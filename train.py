import torch
import torch.utils.tensorboard
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

# sys.path.append('/home/alta/BLTSpeaking/exp-ytl28/local-ytl/grammatical-error-correction/')
from utils.misc import set_global_seeds, print_config, save_config, check_srctgt
from utils.misc import validate_config, get_memory_alloc, convert_dd_att_ref
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor, _del_var
from utils.dataset import Dataset
from utils.config import PAD, EOS
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.recurrent import Seq2Seq

logging.basicConfig(level=logging.DEBUG)
device = torch.device('cpu')

MAX_COUNT_NO_IMPROVE = 5
MAX_COUNT_NUM_ROLLBACK = 5
KEEP_NUM = 1

def load_arguments(parser):

	""" Seq2Seq model """

	# paths
	parser.add_argument('--train_path_src', type=str, required=True, help='train src dir')
	parser.add_argument('--train_path_tgt', type=str, required=True, help='train tgt dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, required=True, help='vocab tgt dir')
	parser.add_argument('--dev_path_src', type=str, default=None, help='dev src dir')
	parser.add_argument('--dev_path_tgt', type=str, default=None, help='dev tgt dir')
	parser.add_argument('--save', type=str, required=True, help='model save dir')
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--load_embedding_src', type=str, default=None, help='pretrained src embedding')
	parser.add_argument('--load_embedding_tgt', type=str, default=None, help='pretrained tgt embedding')

	# model
	parser.add_argument('--use_bpe', type=str, default='False', help='use byte-pair emcoding or not')
	parser.add_argument('--embedding_size_enc', type=int, default=200, help='encoder embedding size')
	parser.add_argument('--embedding_size_dec', type=int, default=200, help='decoder embedding size')
	parser.add_argument('--hidden_size_enc', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--num_bilstm_enc', type=int, default=2, help='number of encoder bilstm layers')
	parser.add_argument('--num_unilstm_enc', type=int, default=0, help='number of encoder unilstm layers')
	parser.add_argument('--hidden_size_dec', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--num_unilstm_dec', type=int, default=2, help='number of encoder bilstm layers')
	parser.add_argument('--hard_att', type=str, default='False', help='use hard attention or not')
	parser.add_argument('--att_mode', type=str, default='bahdanau', \
							help='attention mechanism mode - bahdanau / hybrid / dot_prod')
	parser.add_argument('--hidden_size_att', type=int, default=1, \
							help='hidden size for bahdanau / hybrid attention')
	parser.add_argument('--hidden_size_shared', type=int, default=200, \
							help='transformed att output hidden size (set as hidden_size_enc)')
	parser.add_argument('--additional_key_size', type=int, default=0, \
							help='additional attention key size: keys = [values, add_feats]')

	# train
	parser.add_argument('--random_seed', type=int, default=666, help='random seed')
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--embedding_dropout', type=float, default=0.0, help='embedding dropout')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
	parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0, help='ratio of teacher forcing')
	parser.add_argument('--num_epochs', type=int, default=10, help='number of training epoches')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--residual', type=str, default='False', help='residual connection')
	parser.add_argument('--max_grad_norm', type=float, default=1.0, help='optimiser gradient norm clipping: max grad norm')
	parser.add_argument('--batch_first', type=str, default='True', help='batch as the first dimension')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_with_mask', type=str, default='True', help='calc loss excluding padded words')
	parser.add_argument('--scheduled_sampling', type=str, default='False', \
					 		help='gradually turn off teacher forcing \
					 		(if True, use teacher_forcing_ratio as the starting point)')
	parser.add_argument('--seqrev', type=str, default='False', help='reverse src, tgt sequence')

	# save and print
	parser.add_argument('--checkpoint_every', type=int, default=10, help='save ckpt every n steps')
	parser.add_argument('--print_every', type=int, default=10, help='print every n steps')

	# -- inactive arguments --
	parser.add_argument('--train_attkey_path', type=str, default=None, help='train set additional attention key')
	parser.add_argument('--dev_attkey_path', type=str, default=None, help='dev set additional attention key')
	parser.add_argument('--ptr_net', type=str, default='null', \
							help='whether or not to use pointer network - use attention weights to directly map embedding \
							comb | pure | none: comb combines posterior and att weights (combination ratio is learnt); \
							pure use att weights only; none use posterior only')
	parser.add_argument('--ddatt_loss_weight', type=float, default=0.0, help='determines attloss weight in sgd [0.0~1.0]')
	parser.add_argument('--ddattcls_loss_weight', type=float, default=0.0,
							help='determines attcls loss weight in sgd [0.0~1.0] (att column wise regularisation)')
	parser.add_argument('--att_scale_up', type=float, default=0.0,
							help='scale up att scores before cross entropy loss in regularisation')

	return parser


class Trainer(object):

	def __init__(self, expt_dir='experiment',
		load_dir=None,
		loss=NLLLoss(),
		batch_size=64,
		random_seed=None,
		checkpoint_every=100,
		print_every=100,
		use_gpu=False,
		learning_rate=0.001,
		max_grad_norm=1.0,
		eval_with_mask=True,
		scheduled_sampling=False,
		teacher_forcing_ratio=0.0,
		ddatt_loss_weight=0.0,
		ddattcls_loss_weight=0.0,
		att_scale_up=0.0
		):

		self.random_seed = random_seed
		if random_seed is not None:
			set_global_seeds(random_seed)

		self.loss = loss
		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every
		self.use_gpu = use_gpu
		self.learning_rate = learning_rate
		self.max_grad_norm = max_grad_norm
		self.eval_with_mask = eval_with_mask
		self.scheduled_sampling = scheduled_sampling
		self.teacher_forcing_ratio = teacher_forcing_ratio

		self.ddatt_loss_weight = ddatt_loss_weight
		self.ddattcls_loss_weight = ddattcls_loss_weight
		self.att_scale_up = att_scale_up

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.load_dir = load_dir

		self.batch_size = batch_size
		self.logger = logging.getLogger(__name__)
		self.writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.expt_dir)


	def _evaluate_batches(self, model, batches, dataset):

		model.eval()

		loss = NLLLoss()
		loss.reset()

		match = 0
		total = 0

		out_count = 0
		with torch.no_grad():
			for batch in batches:

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				if 'src_ddfd_probs' in batch and model.additional_key_size > 0:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)
				src_labs = None
				if 'src_ddfd_labs' in batch:
					src_labs =  batch['src_ddfd_labs']
					src_labs = _convert_to_tensor(src_labs, self.use_gpu).unsqueeze(2)

				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)

				non_padding_mask_tgt = tgt_ids.data.ne(PAD)
				non_padding_mask_src = src_ids.data.ne(PAD)

				decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids,
					is_training=False, att_key_feats=src_probs)

				# Evaluation
				logps = torch.stack(decoder_outputs, dim=1).to(device=device)
				if not self.eval_with_mask:
					loss.eval_batch(logps.reshape(-1, logps.size(-1)),
						tgt_ids[:, 1:].reshape(-1))
				else:
					loss.eval_batch_with_mask(logps.reshape(-1, logps.size(-1)),
						tgt_ids[:, 1:].reshape(-1), non_padding_mask_tgt[:,1:].reshape(-1))

				seqlist = other['sequence']
				seqres = torch.stack(seqlist, dim=1).to(device=device)
				correct = seqres.view(-1).eq(tgt_ids[:,1:].reshape(-1))\
					.masked_select(non_padding_mask_tgt[:,1:].reshape(-1)).sum().item()
				match += correct
				total += non_padding_mask_tgt[:,1:].sum().item()

				if not self.eval_with_mask:
					loss.norm_term = 1.0 * tgt_ids.size(0) * tgt_ids[:,1:].size(1)
				else:
					loss.norm_term = 1.0 * torch.sum(non_padding_mask_tgt[:,1:])
				loss.normalise()

				if out_count < 3:
					srcwords = _convert_to_words_batchfirst(src_ids, dataset.tgt_id2word)
					refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], dataset.tgt_id2word)
					seqwords = _convert_to_words(seqlist, dataset.tgt_id2word)
					outsrc = 'SRC: {}\n'.format(' '.join(srcwords[0])).encode('utf-8')
					outref = 'REF: {}\n'.format(' '.join(refwords[0])).encode('utf-8')
					outline = 'GEN: {}\n'.format(' '.join(seqwords[0])).encode('utf-8')
					sys.stdout.buffer.write(outsrc)
					sys.stdout.buffer.write(outref)
					sys.stdout.buffer.write(outline)
					out_count += 1

		att_resloss = 0
		attcls_resloss = 0
		resloss = loss.get_loss()

		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total
		torch.cuda.empty_cache()

		losses = {}
		losses['att_loss'] = att_resloss
		losses['attcls_loss'] = attcls_resloss

		return resloss, accuracy, losses


	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, src_labs=None):

		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = NLLLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			# use self.teacher_forcing_ratio as the starting point
			progress = 1.0 * step / total_steps
			teacher_forcing_ratio = 1.0 - progress

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids,
			is_training=True,teacher_forcing_ratio=teacher_forcing_ratio,att_key_feats=src_probs)

		# Get loss
		loss.reset()
		att_loss.reset()
		dsfclassify_loss.reset()
		# import pdb; pdb.set_trace()

		logps = torch.stack(decoder_outputs, dim=1).to(device=device)
		if not self.eval_with_mask:
			loss.eval_batch(logps.reshape(-1, logps.size(-1)),
				tgt_ids[:, 1:].reshape(-1))
		else:
			loss.eval_batch_with_mask(logps.reshape(-1, logps.size(-1)),
				tgt_ids[:,1:].reshape(-1), non_padding_mask_tgt[:,1:].reshape(-1))

		if not self.eval_with_mask:
			loss.norm_term = 1.0 * tgt_ids.size(0) * tgt_ids[:,1:].size(1)
		else:
			loss.norm_term = 1.0 * torch.sum(non_padding_mask_tgt[:,1:])
		loss.normalise()

		# Backward propagation
		model.zero_grad()
		resloss = loss.get_loss()
		att_resloss = 0
		dsfclassify_resloss = 0

		self.optimizer.step()
		loss.backward()

		return resloss, att_resloss, dsfclassify_resloss


	def _train_epoches(self, train_set, model, n_epochs, start_epoch, start_step, dev_set=None):

		log = self.logger

		print_loss_total = 0  # Reset every print_every
		epoch_loss_total = 0  # Reset every epoch
		att_print_loss_total = 0  # Reset every print_every
		att_epoch_loss_total = 0  # Reset every epoch
		attcls_print_loss_total = 0  # Reset every print_every
		attcls_epoch_loss_total = 0  # Reset every epoch

		step = start_step
		step_elapsed = 0
		prev_acc = 0.0
		count_no_improve = 0
		count_num_rollback = 0
		ckpt = None

		# ******************** [loop over epochs] ********************
		for epoch in range(start_epoch, n_epochs + 1):

			for param_group in self.optimizer.optimizer.param_groups:
				print('epoch:{} lr: {}'.format(epoch, param_group['lr']))
				lr_curr = param_group['lr']

			# ----------construct batches-----------
			# allow re-shuffling of data
			if type(train_set.attkey_path) == type(None):
				print('--- construct train set ---')
				train_batches, vocab_size = train_set.construct_batches(is_train=True)
				if dev_set is not None:
					print('--- construct dev set ---')
					dev_batches, vocab_size = dev_set.construct_batches(is_train=False)
			else:
				print('--- construct train set ---')
				train_batches, vocab_size = train_set.construct_batches_with_ddfd_prob(is_train=True)
				if dev_set is not None:
					print('--- construct dev set ---')
					assert type(dev_set.attkey_path) != type(None), 'Dev set missing ddfd probabilities'
					dev_batches, vocab_size = dev_set.construct_batches_with_ddfd_prob(is_train=False)

			# --------print info for each epoch----------
			steps_per_epoch = len(train_batches)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))


			log.debug(" ----------------- Epoch: %d, Step: %d -----------------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))
			self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# ******************** [loop over batches] ********************
			model.train(True)
			for batch in train_batches:

				# update macro count
				step += 1
				step_elapsed += 1

				# load data
				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']

				src_probs = None
				src_labs = None
				if 'src_ddfd_probs' in batch and model.additional_key_size > 0:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)
				if 'src_ddfd_labs' in batch:
					src_labs = batch['src_ddfd_labs']
					src_labs = _convert_to_tensor(src_labs, self.use_gpu).unsqueeze(2)

				# sanity check src-tgt pair
				if step == 1:
					print('--- Check src tgt pair ---')
					log_msgs = check_srctgt(src_ids, tgt_ids, train_set.src_id2word, train_set.tgt_id2word)
					for log_msg in log_msgs:
						sys.stdout.buffer.write(log_msg)

				# convert variable to tensor
				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)

				# Get loss
				loss, att_loss, attcls_loss = self._train_batch(src_ids, tgt_ids,
					model, step, total_steps, src_probs=src_probs, src_labs=src_labs)

				print_loss_total += loss
				epoch_loss_total += loss
				att_print_loss_total += att_loss
				att_epoch_loss_total += att_loss
				attcls_print_loss_total += attcls_loss
				attcls_epoch_loss_total += attcls_loss

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					att_print_loss_avg = att_print_loss_total / self.print_every
					attcls_print_loss_avg = attcls_print_loss_total / self.print_every
					print_loss_total = 0
					att_print_loss_total = 0
					attcls_print_loss_total = 0


					log_msg = 'Progress: %d%%, Train nlll: %.4f, att: %.4f, attcls: %.4f' % (
								step / total_steps * 100,
								print_loss_avg, att_print_loss_avg, attcls_print_loss_avg)
					# print(log_msg)
					log.info(log_msg)
					self.writer.add_scalar('train_loss', print_loss_avg, global_step=step)
					self.writer.add_scalar('att_train_loss', att_print_loss_avg, global_step=step)
					self.writer.add_scalar('attcls_train_loss', attcls_print_loss_avg, global_step=step)

				# Checkpoint
				if step % self.checkpoint_every == 0 or step == total_steps:

					# save criteria
					if dev_set is not None:
						dev_loss, accuracy, dev_attlosses = \
							self._evaluate_batches(model, dev_batches, dev_set)
						dev_attloss = dev_attlosses['att_loss']
						dev_attclsloss = dev_attlosses['attcls_loss']
						log_msg = 'Progress: %d%%, Dev loss: %.4f, accuracy: %.4f, att: %.4f, attcls: %.4f' % (
									step / total_steps * 100, dev_loss, accuracy,
									dev_attloss, dev_attclsloss)
						log.info(log_msg)
						self.writer.add_scalar('dev_loss', dev_loss, global_step=step)
						self.writer.add_scalar('dev_acc', accuracy, global_step=step)
						self.writer.add_scalar('att_dev_loss', dev_attloss, global_step=step)
						self.writer.add_scalar('attcls_dev_loss', dev_attclsloss, global_step=step)

						# save
						if prev_acc < accuracy:
							# save best model
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=train_set.vocab_src,
									   output_vocab=train_set.vocab_tgt)

							saved_path = ckpt.save(self.expt_dir)
							print('saving at {} ... '.format(saved_path))
							# reset
							prev_acc = accuracy
							count_no_improve = 0
							count_num_rollback = 0
						else:
							count_no_improve += 1

						# roll back
						if count_no_improve > MAX_COUNT_NO_IMPROVE:
							# resuming
							latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
							if type(latest_checkpoint_path) != type(None):
								resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
								print('epoch:{} step: {} - rolling back {} ...'
									.format(epoch, step, latest_checkpoint_path))
								model = resume_checkpoint.model
								self.optimizer = resume_checkpoint.optimizer
								# A walk around to set optimizing parameters properly
								resume_optim = self.optimizer.optimizer
								defaults = resume_optim.param_groups[0]
								defaults.pop('params', None)
								defaults.pop('initial_lr', None)
								self.optimizer.optimizer = resume_optim\
									.__class__(model.parameters(), **defaults)
								# start_epoch = resume_checkpoint.epoch
								# step = resume_checkpoint.step

							# reset
							count_no_improve = 0
							count_num_rollback += 1

						# update learning rate
						if count_num_rollback > MAX_COUNT_NUM_ROLLBACK:

							# roll back
							latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
							if type(latest_checkpoint_path) != type(None):
								resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
								print('epoch:{} step: {} - rolling back {} ...'
									.format(epoch, step, latest_checkpoint_path))
								model = resume_checkpoint.model
								self.optimizer = resume_checkpoint.optimizer
								# A walk around to set optimizing parameters properly
								resume_optim = self.optimizer.optimizer
								defaults = resume_optim.param_groups[0]
								defaults.pop('params', None)
								defaults.pop('initial_lr', None)
								self.optimizer.optimizer = resume_optim\
									.__class__(model.parameters(), **defaults)
								start_epoch = resume_checkpoint.epoch
								step = resume_checkpoint.step

							# decrease lr
							for param_group in self.optimizer.optimizer.param_groups:
								param_group['lr'] *= 0.5
								lr_curr = param_group['lr']
								print('reducing lr ...')
								print('step:{} - lr: {}'.format(step, param_group['lr']))

							# check early stop
							if lr_curr < 0.000125:
								print('early stop ...')
								break

							# reset
							count_no_improve = 0
							count_num_rollback = 0

						model.train(mode=True)
						if ckpt is None:
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=train_set.vocab_src,
									   output_vocab=train_set.vocab_tgt)
							saved_path = ckpt.save(self.expt_dir)
						ckpt.rm_old(self.expt_dir, keep_num=KEEP_NUM)
						print('n_no_improve {}, num_rollback {}'
							.format(count_no_improve, count_num_rollback))
					sys.stdout.flush()

			else:
				if dev_set is None:
					# save every epoch if no dev_set
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
					# saved_path = ckpt.save(self.expt_dir)
					saved_path = ckpt.save_epoch(self.expt_dir, epoch)
					print('saving at {} ... '.format(saved_path))
					continue

				else:
					continue
			# break nested for loop
			break

			if step_elapsed == 0: continue
			epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
			epoch_loss_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)

			log.info('\n')
			log.info(log_msg)


	def train(self, train_set, model, num_epochs=5, resume=False, optimizer=None, dev_set=None):

		"""
			Run training for a given model.
			Args:
				train_set: dataset
				dev_set: dataset, optional
				model: model to run training on, if `resume=True`, it would be
				   overwritten by the model loaded from the latest checkpoint.
				num_epochs (int, optional): number of epochs to run (default 5)
				resume(bool, optional): resume training with the latest checkpoint, (default False)
				optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
				   (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))

			Returns:
				model (seq2seq.models): trained model.
		"""

		log = self.logger.info('MAX_COUNT_NO_IMPROVE: {}'.format(MAX_COUNT_NO_IMPROVE))
		log = self.logger.info('MAX_COUNT_NUM_ROLLBACK: {}'.format(MAX_COUNT_NUM_ROLLBACK))

		torch.cuda.empty_cache()
		if resume:
			# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.load_dir)
			latest_checkpoint_path = self.load_dir
			print('resuming {} ...'.format(latest_checkpoint_path))
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model = resume_checkpoint.model
			print(model)
			self.optimizer = resume_checkpoint.optimizer

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			# start_epoch = resume_checkpoint.epoch
			# step = resume_checkpoint.step
			model.set_idmap(train_set.src_word2id, train_set.src_id2word)
			model.reset_batch_size(train_set.batch_size)

			for name, param in model.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))

			# just for the sake of finetuning
			start_epoch = 1
			step = 0

		else:
			start_epoch = 1
			step = 0
			print(model)

			for name, param in model.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))

			if optimizer is None:
				optimizer = Optimizer(torch.optim.Adam(model.parameters(),
							lr=self.learning_rate), max_grad_norm=self.max_grad_norm) # 5 -> 1
			self.optimizer = optimizer

		self.logger.info("Optimizer: %s, Scheduler: %s"
			% (self.optimizer.optimizer, self.optimizer.scheduler))

		self._train_epoches(train_set, model, num_epochs, start_epoch, step, dev_set=dev_set)

		return model


def main():

	# load config
	parser = argparse.ArgumentParser(description='PyTorch Seq2Seq-DD Training')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# record config
	if not os.path.isabs(config['save']):
		config_save_dir = os.path.join(os.getcwd(), config['save'])
	if not os.path.exists(config['save']):
		os.makedirs(config['save'])

	# check device:
	if config['use_gpu'] and torch.cuda.is_available():
		global device
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('device: {}'.format(device))

	# resume or not
	if config['load']:
		resume = True
		print('resuming {} ...'.format(config['load']))
		config_save_dir = os.path.join(config['save'], 'model-cont.cfg')
	else:
		resume = False
		config_save_dir = os.path.join(config['save'], 'model.cfg')
	save_config(config, config_save_dir)

	# load train set
	train_path_src = config['train_path_src']
	train_path_tgt = config['train_path_tgt']
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	train_attkey_path = config['train_attkey_path']
	train_set = Dataset(train_path_src, train_path_tgt,
						path_vocab_src, path_vocab_tgt,
						attkey_path=train_attkey_path,seqrev=config['seqrev'],
						max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
						use_gpu=config['use_gpu'])

	vocab_size_enc = len(train_set.vocab_src)
	vocab_size_dec = len(train_set.vocab_tgt)

	# load dev set
	if config['dev_path_src'] and config['dev_path_tgt']:
		dev_path_src = config['dev_path_src']
		dev_path_tgt = config['dev_path_tgt']
		dev_attkey_path = config['dev_attkey_path']
		dev_set = Dataset(dev_path_src, dev_path_tgt,
						path_vocab_src, path_vocab_tgt,
						attkey_path=dev_attkey_path,seqrev=config['seqrev'],
						max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
						use_gpu=config['use_gpu'])
	else:
		dev_set = None

	# construct model
	seq2seq = Seq2Seq(vocab_size_enc, vocab_size_dec,
							embedding_size_enc=config['embedding_size_enc'],
							embedding_size_dec=config['embedding_size_dec'],
							embedding_dropout=config['embedding_dropout'],
							hidden_size_enc=config['hidden_size_enc'],
							num_bilstm_enc=config['num_bilstm_enc'],
							num_unilstm_enc=config['num_unilstm_enc'],
							hidden_size_dec=config['hidden_size_dec'],
							num_unilstm_dec=config['num_unilstm_dec'],
							hidden_size_att=config['hidden_size_att'],
							hidden_size_shared=config['hidden_size_shared'],
							dropout=config['dropout'],
							residual=config['residual'],
							batch_first=config['batch_first'],
							max_seq_len=config['max_seq_len'],
							batch_size=config['batch_size'],
							load_embedding_src=config['load_embedding_src'],
							load_embedding_tgt=config['load_embedding_tgt'],
							src_word2id=train_set.src_word2id,
							tgt_word2id=train_set.tgt_word2id,
							src_id2word=train_set.src_id2word,
							att_mode=config['att_mode'],
							hard_att=config['hard_att'],
							use_gpu=config['use_gpu'],
							additional_key_size=config['additional_key_size'],
							ptr_net=config['ptr_net'],
							use_bpe=config['use_bpe'])

	if config['use_gpu']:
		seq2seq = seq2seq.cuda()

	# contruct trainer
	t = Trainer(expt_dir=config['save'],
					load_dir=config['load'],
					batch_size=config['batch_size'],
					random_seed=config['random_seed'],
					checkpoint_every=config['checkpoint_every'],
					print_every=config['print_every'],
					learning_rate=config['learning_rate'],
					eval_with_mask=config['eval_with_mask'],
					scheduled_sampling=config['scheduled_sampling'],
					teacher_forcing_ratio=config['teacher_forcing_ratio'],
					use_gpu=config['use_gpu'],
					max_grad_norm=config['max_grad_norm'],
					ddatt_loss_weight=config['ddatt_loss_weight'],
					ddattcls_loss_weight=config['ddattcls_loss_weight'],
					att_scale_up=config['att_scale_up'])

	# run training
	seq2seq = t.train(train_set, seq2seq,
		num_epochs=config['num_epochs'], resume=resume, dev_set=dev_set)


if __name__ == '__main__':
	main()

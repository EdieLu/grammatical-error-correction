import torch
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

# sys.path.append('/home/alta/BLTSpeaking/exp-ytl28/local-ytl/grammatical-error-correction/')
from utils.dataset import Dataset
from utils.misc import set_global_seeds, print_config, save_config
from utils.misc import validate_config, get_memory_alloc
from utils.misc import _convert_to_words_batchfirst, _convert_to_words
from utils.misc import _convert_to_tensor, plot_alignment
from utils.config import PAD, EOS
from modules.loss import NLLLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.recurrent import Seq2Seq

import logging
logging.basicConfig(level=logging.INFO)

device = torch.device('cpu')

def load_arguments(parser):

	""" Seq2Seq eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--test_path_tgt', type=str, required=True, help='test tgt dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, required=True, help='vocab tgt dir')
	parser.add_argument('--load', type=str, required=True, help='model load dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')
	parser.add_argument('--test_attkey_path', type=str, default=None, help='test set additional attention key')

	# others
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--beam_width', type=int, default=0, help='beam width; set to 0 to disable beam search')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_mode', type=int, default=2, help='which evaluation mode to use')
	parser.add_argument('--seqrev', type=str, default=False, help='whether or not to reverse sequence')

	return parser


def translate(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width, seqrev=False):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model.to(device)
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)
	model.reset_batch_size(test_set.batch_size)
	model.check_var('ptr_net')
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	if type(test_set.attkey_path) == type(None):
		test_batches, vocab_size = test_set.construct_batches(is_train=False)
	else:
		test_batches, vocab_size = test_set.construct_batches_with_ddfd_prob(is_train=False)

	# f = open(os.path.join(test_path_out, 'translate.txt'), 'w') -> use proper encoding
	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		match = 0
		total = 0
		with torch.no_grad():
			for batch in test_batches:

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				src_probs = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)

				src_ids = _convert_to_tensor(src_ids, use_gpu)
				decoder_outputs, decoder_hidden, other = model(src=src_ids,
																is_training=False,
																att_key_feats=src_probs,
																beam_width=beam_width)
				# memory usage
				mem_kb, mem_mb, mem_gb = get_memory_alloc()
				mem_mb = round(mem_mb, 2)
				print('Memory used: {0:.2f} MB'.format(mem_mb))

				# write to file
				seqlist = other['sequence']
				seqwords = _convert_to_words(seqlist, test_set.src_id2word)
				for i in range(len(seqwords)):
					# skip padding sentences in batch (num_sent % batch_size != 0)
					if src_lengths[i] == 0:
						continue
					words = []
					for word in seqwords[i]:
						if word == '<pad>':
							continue
						elif word == '</s>':
							break
						else:
							words.append(word)
					if len(words) == 0:
						outline = ''
					else:
						if seqrev:
							words = words[::-1]
						outline = ' '.join(words)
					f.write('{}\n'.format(outline))
					# if i == 0:
					# 	print(outline)
				sys.stdout.flush()


def evaluate(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width, seqrev=False):

	"""
		with reference tgt given - Run translation.
		Args:
			test_set: test dataset
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
		Returns:
			accuracy (excluding PAD tokens)
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model.to(device)
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)
	model.reset_batch_size(test_set.batch_size)
	model.set_beam_width(beam_width)
	model.check_var('ptr_net')
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	if type(test_set.attkey_path) == type(None):
		test_batches, vocab_size = test_set.construct_batches(is_train=False)
	else:
		test_batches, vocab_size = test_set.construct_batches_with_ddfd_prob(is_train=False)


	# f = open(os.path.join(test_path_out, 'test.txt'), 'w')
	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		match = 0
		total = 0
		with torch.no_grad():
			for batch in test_batches:

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)

				src_ids = _convert_to_tensor(src_ids, use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)

				decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids,
																is_training=False,
																att_key_feats=src_probs,
																beam_width=beam_width)

				# Evaluation
				seqlist = other['sequence'] # traverse over time not batch
				if beam_width > 1:
					full_seqlist = other['topk_sequence']
					decoder_outputs = decoder_outputs[:-1]
				for step, step_output in enumerate(decoder_outputs):
					target = tgt_ids[:, step+1]
					non_padding = target.ne(PAD)
					correct = seqlist[step].view(-1).eq(target)
						.masked_select(non_padding).sum().item()
					match += correct
					total += non_padding.sum().item()

				# write to file
				seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
				for i in range(len(seqwords)):
					# skip padding sentences in batch (num_sent % batch_size != 0)
					if src_lengths[i] == 0:
						continue
					words = []
					for word in seqwords[i]:
						if word == '<pad>':
							continue
						elif word == '</s>':
							break
						else:
							words.append(word)
					if len(words) == 0:
						outline = ''
					else:
						if seqrev:
							words = words[::-1]
						outline = ' '.join(words)
					f.write('{}\n'.format(outline))
					if i == 0:
						print(outline)
				sys.stdout.flush()

		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total

	return accuracy


def att_plot(test_set, load_dir, plot_path, use_gpu, max_seq_len, beam_width):

	"""
		generate attention alignment plots
		Args:
			test_set: test dataset
			load_dir: model dir
			use_gpu: on gpu/cpu
			max_seq_len
		Returns:

	"""

	# check devide
	print('cuda available: {}'.format(torch.cuda.is_available()))
	use_gpu = use_gpu and torch.cuda.is_available()

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model.to(device)
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)
	model.reset_batch_size(test_set.batch_size)

	# in plotting mode always turn off beam search
	model.set_beam_width(beam_width=0)
	model.check_var('ptr_net')
	print('max seq len {}'.format(model.max_seq_len))
	print('ptr_net {}'.format(model.ptr_net))

	# load test
	if type(test_set.attkey_path) == type(None):
		test_batches, vocab_size = test_set.construct_batches(is_train=False)
	else:
		test_batches, vocab_size = test_set.construct_batches_with_ddfd_prob(is_train=False)

	# start eval
	model.eval()
	match = 0
	total = 0
	count = 0
	with torch.no_grad():
		for batch in test_batches:

			src_ids = batch['src_word_ids']
			src_lengths = batch['src_sentence_lengths']
			tgt_ids = batch['tgt_word_ids']
			tgt_lengths = batch['tgt_sentence_lengths']
			src_probs = None
			if 'src_ddfd_probs' in batch:
				src_probs =  batch['src_ddfd_probs']
				src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)

			src_ids = _convert_to_tensor(src_ids, use_gpu)
			tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)

			decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids,
															is_training=False,
															att_key_feats=src_probs,
															beam_width=0)
			# Evaluation
			# default batch_size = 1
			# attention: 31 * [1 x 1 x 32] ( tgt_len(query_len) * [ batch_size x 1 x src_len(key_len)] )
			attention = other['attention_score']
			seqlist = other['sequence'] # traverse over time not batch
			bsize = test_set.batch_size
			max_seq = test_set.max_seq_len
			vocab_size = len(test_set.tgt_word2id)
			for idx in range(len(decoder_outputs)): # loop over max_seq
				step = idx
				step_output = decoder_outputs[idx] # 64 x vocab_size
				# count correct
				target = tgt_ids[:, step+1]
				non_padding = target.ne(PAD)
				correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
				match += correct
				total += non_padding.sum().item()

			# Print sentence by sentence
			srcwords = _convert_to_words_batchfirst(src_ids, test_set.src_id2word)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], test_set.tgt_id2word)
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
			# print(type(attention))
			# print(len(attention))
			# print(type(attention[0]))
			# print(attention[0].size())
			# input('...')
			n_q = len(attention)
			n_k = attention[0].size(2)
			b_size =  attention[0].size(0)
			att_score = torch.empty(n_q, n_k, dtype=torch.float)
			# att_score = np.empty([n_q, n_k])

			for i in range(len(seqwords)): # loop over sentences
				outline_src = ' '.join(srcwords[i])
				outline_ref = ' '.join(refwords[i])
				outline_gen = ' '.join(seqwords[i])
				print('SRC: {}'.format(outline_src))
				print('REF: {}'.format(outline_ref))
				print('GEN: {}'.format(outline_gen))
				for j in range(len(attention)):
					# i: idx of batch
					# j: idx of query
					gen = seqwords[i][j]
					ref = refwords[i][j]
					att = attention[j][i]
					# record att scores
					att_score[j] = att

					# print('REF:GEN - {}:{}'.format(ref,gen))
					# print('{}th ATT size: {}'.format(j, attention[j][i].size()))
					# print(att)
					# print(torch.argmax(att))
					# print(sum(sum(att)))
					# input('Press enter to continue ...')

				# plotting
				# print(att_score)
				loc_eos_k = srcwords[i].index('</s>') + 1
				loc_eos_q = seqwords[i].index('</s>') + 1
				loc_eos_ref = refwords[i].index('</s>') + 1
				print('eos_k: {}, eos_q: {}'.format(loc_eos_k, loc_eos_q))
				att_score_trim = att_score[:loc_eos_q, :loc_eos_k] # each row (each query) sum up to 1
				print(att_score_trim)
				print('\n')
				# import pdb; pdb.set_trace()

				choice = input('Plot or not ? - y/n\n')
				if choice:
					if choice.lower()[0] == 'y':
						print('plotting ...')
						plot_dir = os.path.join(plot_path, '{}.png'.format(count))
						src = srcwords[i][:loc_eos_k]
						hyp = seqwords[i][:loc_eos_q]
						ref = refwords[i][:loc_eos_ref]
						# x-axis: src; y-axis: hyp
						# plot_alignment(att_score_trim.numpy(), plot_dir, src=src, hyp=hyp, ref=ref)
						plot_alignment(att_score_trim.numpy(), plot_dir, src=src, hyp=hyp, ref=None) # no ref
						count += 1
						input('Press enter to continue ...')


	if total == 0:
		accuracy = float('nan')
	else:
		accuracy = match / total
	print(saccuracy)


def main():

	# load config
	parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)
	config_save_dir = os.path.join(config['load'], 'eval.cfg')
	save_config(config, config_save_dir)

	# check device:
	if config['use_gpu'] and torch.cuda.is_available():
		global device
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('device: {}'.format(device))

	# load src-tgt pair
	test_path_src = config['test_path_src']
	test_path_tgt = config['test_path_tgt']
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	test_path_out = config['test_path_out']
	test_attkey_path = config['test_attkey_path']
	load_dir = config['load']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	seqrev = config['seqrev']
	print('attkey dir: {}'.format(test_attkey_path))
	print('reverse seq: {}'.format(seqrev))
	print('use gpu: {}'.format(use_gpu))

	if not os.path.exists(test_path_out):
		os.makedirs(test_path_out)

	if not os.path.exists(test_path_out):
		os.makedirs(test_path_out)
	config_save_dir = os.path.join(test_path_out, 'eval.cfg')
	save_config(config, config_save_dir)

	# set test mode
	MODE = config['eval_mode']
	if MODE == 3:
		max_seq_len = 32
		batch_size = 1
		beam_width = 1
		use_gpu = False

	# load test_set
	test_set = Dataset(test_path_src, test_path_tgt,
						path_vocab_src, path_vocab_tgt,
						attkey_path=test_attkey_path,seqrev=seqrev,
						max_seq_len=max_seq_len, batch_size=batch_size,
						use_gpu=use_gpu)
	print('Testset loaded')
	sys.stdout.flush()

	# run eval
	if MODE == 1:
		# run evaluation
		# print("use gpu: {}".format(config['use_gpu']))
		accuracy = evaluate(test_set, load_dir, test_path_out,
			use_gpu, max_seq_len, beam_width, seqrev=seqrev)
		print(accuracy)

	elif MODE == 2:
		translate(test_set, load_dir, test_path_out,
			use_gpu, max_seq_len, beam_width, seqrev=seqrev)

	elif MODE == 3:
		# plotting
		att_plot(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width)




if __name__ == '__main__':
	main()

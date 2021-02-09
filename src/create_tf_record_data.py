"""Script for pre-processing raw data"""

import logging
import argparse
from data_preprocessing import TriLetterExtractor, process_datasets_to_file
from tokenizer import TwinBertTokenizer
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_data_path', type=str, default="../../../data/QK_Graph/QK_ANN_Neighbor_Teacher.tsv", help="Path to teacher data")
parser.add_argument('--teacher_data_save_path', type=str, default="../data/QK_ANN_Neighbor/Teacher/", help="Path to save teacher data")
parser.add_argument('--teacher_eval_data_path', type=str, default="../../../data/QK_Graph/QK_ANN_Neighbor_Teacher.tsv", help="Path to teacher eval data")
parser.add_argument('--teacher_eval_data_save_path', type=str, default="../data/QK_ANN_Neighbor/Teacher_Eval/", help="Path to save teacher eval data")
parser.add_argument('--validation_data_path', type=str, default="../../../data/QK_Graph/QK_ANN_Neighbor_Validation.tsv", help="Path to validation data")
parser.add_argument('--validation_data_save_path', type=str, default="../data/QK_ANN_Neighbor/Validation/", help="Path to save validation data")
parser.add_argument('--vocab_path', type=str, default="../config/l3g.txt", help="Path to vocab file, only used by triletter tokenizer")
parser.add_argument('--chunksize', type=int, default=1e6, help="Pandas loading chunksize")
parser.add_argument('--skip_chunk', type=int, default=0, help="Number of chunks to skip")
parser.add_argument('--n_chunk', type=int, default=0, help="Number of chunks to process")
parser.add_argument('--tokenizer_type', type=str, default="bpe", help="Tokenizer type")
parser.add_argument('--max_n_letters', type=int, default=20, help="Only used by triletter tokenizer")
parser.add_argument('--max_seq_len', type=int, default=16, help="Max length of sequence")
parser.add_argument('--tokenizer_name', type=str, default="bert-base-uncased", help="Pre-trained Bert tokenizer name")
parser.add_argument('--a_fanouts', type=str, default="3", help="a fanouts")
parser.add_argument('--b_fanouts', type=str, default="3", help="b fanouts")
args = parser.parse_args()


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.tokenizer_type == "triletter":
        extractor = TriLetterExtractor(args.vocab_path)
    else:
        extractor = BertTokenizer.from_pretrained(args.tokenizer_name)

    try:
        process_datasets_to_file(args.teacher_data_path, extractor, args.teacher_data_save_path, max_seq_len=args.max_seq_len, max_n_letters=args.max_n_letters, int_label=False, chunksize=args.chunksize, a_fanouts=list(map(int, args.a_fanouts.split(","))), b_fanouts=list(map(int, args.b_fanouts.split(","))), skip_chunk=args.skip_chunk, n_chunk=args.n_chunk)
    except Exception as e:
        logger.info(e)
        logger.info("Cannot load from raw teacher data")


    try:
        process_datasets_to_file(args.teacher_eval_data_path, extractor, args.teacher_eval_data_save_path, max_seq_len=args.max_seq_len, max_n_letters=args.max_n_letters, int_label=False, chunksize=args.chunksize, top=1000000, convert_to_int=True, a_fanouts=list(map(int, args.a_fanouts.split(","))), b_fanouts=list(map(int, args.b_fanouts.split(","))), n_chunk=args.n_chunk)
    except Exception as e:
        logger.info(e)
        logger.info("Cannot load from raw teacher eval data")


    try:
        print("Start processing validation data")
        process_datasets_to_file(args.validation_data_path, extractor, args.validation_data_save_path, max_seq_len=args.max_seq_len, max_n_letters=args.max_n_letters, int_label=True, chunksize=args.chunksize, a_fanouts=list(map(int, args.a_fanouts.split(","))), b_fanouts=list(map(int, args.b_fanouts.split(","))))
    except Exception as e:
        logger.info(e)
        logger.info("Cannot load from raw validation data")


if __name__ == "__main__":
    main()

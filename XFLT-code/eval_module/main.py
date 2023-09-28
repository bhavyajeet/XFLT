from metric_calculator import MetricCalculator
from metric_calculator import ParentCalculator
import sys
import argparse
from indicnlp.tokenize import indic_tokenize
import glob

class Tokenizer:
  """Abstract base class for a tokenizer.
  Subclasses of Tokenizer must implement the tokenize() method.
  """
  def tokenize(self, text):
    return text


parser = argparse.ArgumentParser(description='PARENT calculator')
parser.add_argument('--root_file', type=str, help='root file')
parser.add_argument('--test_path', type=str, help='test path')
parser.add_argument('--single', type=int, help='sentence level or document level')
parser.add_argument('--exclusion', type=int, default=0, help='exclude less than x sentences for computation')
parser.add_argument('--parent', type=int, default=0, help='parent')
parser.add_argument('--threshold', default='lang', type=str)
parser.add_argument('--name', type=str)
args = parser.parse_args()

root_file = args.root_file
test_json_path = args.test_path
single = bool(int(args.single))
exclusion = int(args.exclusion)
parent = bool(int(args.parent))
threshold_type = args.threshold
name = args.name

print(root_file, test_json_path, single, exclusion, parent, threshold_type, name)
tokenizer_obj = Tokenizer()
tokenizer_obj.tokenize = indic_tokenize.trivial_tokenize

gen_file, ref_file, src_file = sorted(glob.glob(f'{root_file}/*.txt'))
print(gen_file, ref_file, src_file)
if(not(parent)):
  metric_calc = MetricCalculator(src_file, ref_file, gen_file, test_json_path, tokenizer_obj, root_file, single=single, exclusion=exclusion, name=name)
  metric_calc.get_all_scores(root_file)
else:
  parent_calc = ParentCalculator(src_file, ref_file, gen_file, test_json_path, tokenizer_obj, root_file, single=single, exclusion=exclusion, name=name, threshold_type=threshold_type)
  if(exclusion == 0):
    parent_calc.compute_xparent(root_file)
  parent_calc.combine_xparent(root_file, trade_off=0.5)

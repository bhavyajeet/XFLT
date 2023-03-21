from metric_calculator import MetricCalculator, ParentCalculator
import sys 
from indicnlp.tokenize import indic_tokenize
import glob 

root_file = sys.argv[1]
single = bool(int(sys.argv[2]))

class Tokenizer:
  """Abstract base class for a tokenizer.
  Subclasses of Tokenizer must implement the tokenize() method.
  """
  def tokenize(self, text):
    return text

tokenizer_obj = Tokenizer()
tokenizer_obj.tokenize = indic_tokenize.trivial_tokenize

gen_file, ref_file, src_file = sorted(glob.glob(f'{root_file}/*.txt'))
print(gen_file, ref_file, src_file)
#metric_calc = MetricCalculator(src_file, ref_file, gen_file, f'{root_file}/test_complete.jsonl', tokenizer_obj, single=single)
parent_calc = ParentCalculator(src_file, ref_file, gen_file, f'{root_file}/test_complete.jsonl', tokenizer_obj, single=single)

#metric_calc.get_all_scores(root_file)
parent_calc.compute_xparent(root_file)
parent_calc.combine_xparent(root_file, trade_off=0.5)

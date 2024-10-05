import os
import pdb
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.evaluate.utils.NQ_evaluate_predictions import NQ_evaluate

import json
import os.path
import re




result_file_path = sys.argv[1]

num_correct, num_total, accuracy = NQ_evaluate(result_file_path, result_file_path)

print('{:.2f}'.format(accuracy * 100))
import re
import timeout_decorator
import json
import tree_sitter_python as tspython
from nltk.tokenize import RegexpTokenizer
from tree_sitter import Language, Parser

IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
REGEX_TEXT = ("(?<=[a-z0-9])(?=[A-Z])|"
              "(?<=[A-Z0-9])(?=[A-Z][a-z])|"
              "(?<=[0-9])(?=[a-zA-Z])|"
              "(?<=[A-Za-z])(?=[0-9])|"
              "(?<=[@$.'\"])(?=[a-zA-Z0-9])|"
              "(?<=[a-zA-Z0-9])(?=[@$.'\"])|"
              "_|\\s+")
string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''

code_tokenizer = RegexpTokenizer(r'\w+')

parser = None

@timeout_decorator.timeout(5)
def get_ast(parser, code):
    assert isinstance(code, str) or isinstance(code, bytes)
    if isinstance(code, str):
        code = bytes(code, "utf8")
    try:
        tree = parser.parse(code)
        return tree
    except Exception as e:
        return None

def is_parse_valid(parser, code):
    def syntax_error(node):
        if node.type == "ERROR":
            return True
        try:
            for child in node.children:
                if syntax_error(child):
                    return True
        except RecursionError as err:
            return True

        return False

    tree = get_ast(parser, code)
    if tree is not None:
        return not syntax_error(tree.root_node)
    return False

def get_python_one_statement(prompt, completion:str, parser):
    for i in range(len(completion)):
        code = prompt + completion[:i + 1]
        if not is_parse_valid(parser, code):
            continue
        # if completion[i + 1] == "\n" and not completion[:i+1].isspace():
        if completion[i + 1] == "\n":
            return completion[:i + 1].rstrip()

    return completion

def postprocess_code_lines(prompt, completion, parser, lang):
    try:
        if lang == "python":
            return get_python_one_statement(prompt, completion, parser)
        # if lang in ["java", "csharp", "typescript"]:
        #     return get_bracket_lang_statement(completion)
    except Exception as e:
        return completion

def remove_comments(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'//.*', '', code)
    return code

def process_examples(lang, args):
    sample, ex = args
    
    global parser
    if parser is None:
        if lang == "python":
            PY_LANGUAGE = Language(tspython.language())
            parser = Parser(PY_LANGUAGE)
        else:
            raise Exception("unsupport language in compute_metric_stmt")

    prediction = postprocess_code_lines(ex["prompt"], sample["pred"], parser, lang)
    # prediction = remove_comments(prediction)
    target = ex["groundtruth"]
    target = remove_comments(target)

    pred_lines = [l.strip() for l in prediction.split("\n") if l.strip()]
    gt_lines = [l.strip() for l in target.split("\n") if l.strip()]
    em_label = int(pred_lines == gt_lines)

    trunc_s = {
        "task_id": sample["task_id"],
        "pred": prediction,
        "target": target,
    }
    return trunc_s, em_label


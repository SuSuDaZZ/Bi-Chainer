import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM



def get_condition_consequent(statement, condition_consequenct_examples):
    condition_consequent_prompt = "Identify the condition and consequence the statement."
    condition_consequent_input = condition_consequent_prompt + "\nstatement:\n" + statement 
    condition_consequent_input = condition_consequent_input + condition_consequenct_examples
    return condition_consequent_input


def get_fact_check_input(hypothesis, facts, fact_check_examples):
    fact_check_prompt = "Given a set of premises, you have to reason whether the hypothesis is true, false, or unknown. To prove the hypothesis, you need to check the premises whether the hypothesis can be directly proved or disproved by one of the premises."
    fact_check_input = fact_check_prompt + "\nHypothesis:\n" + hypothesis + "\nPremises:\n" + facts
    fact_check_input = fact_check_input + fact_check_examples
    return fact_check_input


def get_confusion_check_input(resoning_results, confusion_check_examples):
    # confusion_check_prompt = "Check whether each reasoning step produces consistent deduction or induction results after applying the selected rules"
    # LLaMA
    confusion_check_prompt = "Check whether the reasoning results are consistent with each other."
    confusion_check_input = confusion_check_prompt + "\Reasoning Results:\n" + resoning_results
    fact_check_input = confusion_check_input + confusion_check_examples
    return confusion_check_input


def get_fact_identify_input(hypothesis, facts, fact_identify_examples):
    # fact_identify_prompt = "To prove the hypothesis, you need to identify the premises where new conclusions can be derived toward proving the goal."
    # for LLaMA
    fact_identify_prompt = "You need to identify the premises that are related to the condition or the consequence of the hypothesis."
    fact_identify_input = fact_identify_prompt + "\nHypothesis:\n" + hypothesis + "\nPremises:\n" + facts
    fact_identify_input = fact_identify_input + "\nExamples:\n" + fact_identify_examples
    return fact_identify_input


def get_forward_rule_selection_input(hypothesis, identified_facts, rules, forward_rule_selection_examples):
    rule_selection_prompt = "To prove the hypothesis, you need to select the rules whose conditions entail the identified facts and whose consequents entail the consequent of the hypothesis. If a rule satisfying these criteria is found, return it as the result. Otherwise, return only the rules that are entailed by the identified facts."
    rule_selection_input = rule_selection_prompt + + "\nHypothesis:\n" + hypothesis + "\Identified Facts:\n" + identified_facts + "\nRules:\n" + rules
    rule_selection_input = rule_selection_input + "\nExamples:\n" + forward_rule_selection_examples
    return rule_selection_input


def get_backward_rule_selection_input(hypothesis, rules, backward_rule_selection_examples):
    rule_selection_prompt = "To prove the hypothesis, you need to select the rules whose consequences entail the identified facts and whose consequents entail the consequent of the hypothesis."
    rule_selection_input = rule_selection_prompt + + "\nHypothesis:\n" + hypothesis + "\nRules:\n" + rules
    rule_selection_input = rule_selection_input + "\nExamples:\n" + backward_rule_selection_examples
    return rule_selection_input


def get_logical_deduction_input(identified_facts, selected_rules, logical_deduction_examples):
    logical_deduction_prompt = "Derive the inferences based on the selected rules."
    logical_deduction_input = logical_deduction_prompt + "\Identified Facts:\n" + identified_facts + "\nSelected Rules:\n" + selected_rules
    logical_deduction_input = logical_deduction_input + "\nExamples:\n" + logical_deduction_examples
    return logical_deduction_input


def get_logical_abduction_input(hypothesis, selected_rule, logical_abduction_examples):
    logical_abduction_prompt = "Analyze the plausible explanations for the selected rules."
    logical_abduction_input = logical_abduction_prompt + "\Hypothesis:\n" + hypothesis + "\nSelected Rules:\n" + selected_rule
    logical_abduction_input = logical_abduction_input + "\nExamples:\n" + logical_abduction_examples
    return logical_abduction_input

def do_fact_check(hypothesis, facts, fact_check_examples, tokenizer, model, device):
    fact_check_input = get_fact_check_input(hypothesis, facts, fact_check_examples)
    fact_check_output = get_model_output(fact_check_input, tokenizer, model, device)
    print("\nFact Check Result:")
    print(fact_check_output)
    fact_check_output = fact_check_output.lower()
    true_pos = fact_check_output.find("true")
    false_pos = fact_check_output.find("false")

    if true_pos < false_pos:
        fact_check_result = "True"
    elif false_pos < true_pos:
        fact_check_result = "False"
    else:
        fact_check_result = "Unknown"
    return fact_check_result


def do_confusion_check(resoning_results, confusion_check_examples, tokenizer, model, device):
    confusion_check_input = get_confusion_check_input(resoning_results, confusion_check_examples)
    confusion_check_output = get_model_output(confusion_check_input, tokenizer, model, device)
    print("\Confusion Check Result:")
    print(confusion_check_output)
    confusion_check_output = confusion_check_output.lower()
    if "yes" in confusion_check_output:
        confusion_check_result = True
    elif "no" in confusion_check_output:
        confusion_check_result = False
    return confusion_check_result


def do_fact_identify(hypothesis, facts, fact_identify_examples, tokenizer, model, device):
    fact_identify_input = get_fact_identify_input(hypothesis, facts, fact_identify_examples)
    fact_identify_output = get_model_output(fact_identify_input, tokenizer, model, device)
    print("\nFact Identify Result:")
    print(fact_identify_output)
    pattern = r"fact\s+(\d{1,2})"
    matches = re.findall(pattern, fact_identify_output)
    identified_facts = [int(match) for match in matches]
    return identified_facts


def do_forward_rule_selection(hypothesis, rules, identified_facts, forward_rule_selection_examples, tokenizer, model, device):
    rule_selection_input = get_forward_rule_selection_input(hypothesis, identified_facts, rules, forward_rule_selection_examples)
    rule_selection_output = get_model_output(rule_selection_input, tokenizer, model, device)
    print("\nRule Seletion Result:")
    print(rule_selection_output)
    pattern = r"rule\s+(\d{1,2})"
    matches = re.findall(pattern, rule_selection_output)
    selected_rules = [int(match) for match in matches]
    return selected_rules


def do_logical_deduction(identified_facts, selected_rules, logical_deduction_examples, tokenizer, model, device):
    logical_deduction_input = get_logical_deduction_input(identified_facts, selected_rules, logical_deduction_examples)
    deduction = get_model_output(logical_deduction_input, tokenizer, model, device)
    print("\nDeduction Result:")
    print(deduction)
    pattern = r'[^\.\?!;\n]+'
    match = re.search(pattern, deduction)
    deduction = match.group().strip()
    return deduction


def do_logical_abduction(hypothesis, selected_rule, logical_abduction_examples, tokenizer, model, device):
    logical_abduction_input = get_logical_abduction_input(hypothesis, selected_rule, logical_abduction_examples)
    abduction = get_model_output(logical_abduction_input, tokenizer, model, device)
    print("\nAbduction Result:")
    print(abduction)
    pattern = r'[^\.\?!;\n]+'
    match = re.search(pattern, abduction)
    abduction = match.group().strip()
    return abduction


def do_backward_rule_selection(hypothesis, rules, backward_rule_selection_examples, tokenizer, model, device):
    rule_selection_input = get_backward_rule_selection_input(hypothesis, rules, backward_rule_selection_examples)
    rule_selection_output = get_model_output(rule_selection_input, tokenizer, model, device)
    print("\nRule Seletion Result:")
    print(rule_selection_output)
    pattern = r"rule\s+(\d{1,2})"
    matches = re.findall(pattern, rule_selection_output)
    selected_rules = [int(match) for match in matches]
    return selected_rules


def one_condition_consequence_identify(statement, condition_consequenct_examples, tokenizer, model, device):
    statement_input = get_condition_consequent(statement, condition_consequenct_examples)
    statement_output = get_model_output(statement_input, tokenizer, model, device)
    statement_content = statement_output.split("\n\n")[1]

    for line in statement_content.split("\n"):
        if "condition: " in line:
            condition = line.split("condition: ")[1]
        elif "consequence: " in line:
            consequence = line.split("consequence: ")[1]
    
    return condition, consequence
            


def do_condition_consequence_identify(hypothesis, facts, rules, tokenizer, model, device):
    condition_consequences = dict()
    # Hypothesis:
    condition_consequences["hypothesis"] = dict()
    condition, consequence = one_condition_consequence_identify(hypothesis, tokenizer, model, device)
    condition_consequences["hypothesis"]["condition"] = condition
    condition_consequences["hypothesis"]["consequence"] = consequence
    
    # Facts:
    condition_consequences["facts"] = []
    for fact in facts:
        condition, consequence = one_condition_consequence_identify(fact, tokenizer, model, device)
        condition_consequences["facts"].append({"condition": condition, "consequence": consequence})

    # Rules:
    condition_consequences["rules"] = []
    for rule in rules:
        condition, consequence = one_condition_consequence_identify(rule, tokenizer, model, device)
        condition_consequences["rules"].append({"condition": condition, "consequence": consequence})

    print("\nCondition Consequence Result:")
    print(condition_consequences)
    return condition_consequences



def get_model_output(args, model_input, tokenizer, model, device):

    model_inputs = tokenizer(model_input, return_tensors="pt").to(device)

    generated_ids = model.generate(**model_inputs, do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens)
    decoded = tokenizer.batch_decode(generated_ids)
    output = decoded[0]

    return output


def load_model(args):
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, return_dict=True, low_cpu_mem_usage=True)

    model.to(args.device)

    return model, tokenizer


def data_process(data):

    fact_texts = data["fact_texts"]
    rule_texts = data["rule_texts"]
    question_text = data["question_text"]

    hypothesis = question_text

    facts_text = []
    for fact_id, fact in enumerate(fact_texts):
        num = str(fact_id+1)
        context_text = num + ". " + fact.strip()
        facts_text.append(context_text)
    facts = "\n".join(facts_text)

    rules_text = []
    for rule_id, rule in enumerate(rule_texts):
        num = str(rule_id+1)
        context_text = num + ". " + rule.strip()
        rules_text.append(context_text)
    rules = "\n".join(rules_text)

    return hypothesis, facts, rules


def one_forward_chaining(step, hypothesis, facts, rules, identified_facts, fact_check_examples, forward_rule_selection_examples, logical_deduction_examples, confusion_check_examples, tokenizer, model, device):
    direction_flag = "forward"
    selected_rules = do_forward_rule_selection(hypothesis, rules, identified_facts, forward_rule_selection_examples, tokenizer, model, device)
    step += 1
    
    forward_deductions = []

    if len(selected_rules) > 0:

        for selected_rule in selected_rules:
            deduction = do_logical_deduction(identified_facts, selected_rule, logical_deduction_examples, tokenizer, model, device)
            forward_deductions.append(deduction)
            step += 1
            facts.append(deduction)
            identified_facts.append(deduction)

            fact_check_result = do_fact_check(hypothesis, facts, fact_check_examples, tokenizer, model, device)
            step += 1

            if fact_check_result == "Unknown":
                confusion_check_result = do_confusion_check(forward_deductions, confusion_check_examples, tokenizer, model, device)
                step += 1
                if confusion_check_result:
                    direction_flag = "backward"

    else:
        direction_flag = "backward"

    return step, fact_check_result, direction_flag, facts, identified_facts


def one_backward_chaining(step, hypothesis, facts, rules, fact_check_examples, backward_rule_selection_examples, logical_abduction_examples, confusion_check_examples, tokenizer, model, device):
    direction_flag = "backward"
    selected_rules = do_backward_rule_selection(hypothesis, rules, backward_rule_selection_examples, tokenizer, model, device)
    step += 1
    
    backward_abductions = []

    if len(selected_rules) > 0:

        for selected_rule in selected_rules:
            abduction = do_logical_abduction(hypothesis, selected_rule, logical_abduction_examples, tokenizer, model, device)
            backward_abductions.append(abduction)
            step += 1

            hypothesis = "or".join(backward_abductions)

            fact_check_result = do_fact_check(hypothesis, facts, fact_check_examples, tokenizer, model, device)
            step += 1

            if fact_check_result == "Unknown":
                confusion_check_result = do_confusion_check(backward_abductions, confusion_check_examples, tokenizer, model, device)
                step += 1
                if confusion_check_result:
                    direction_flag = "forward"

    else:
        direction_flag = "forward"

    return step, fact_check_result, direction_flag, facts, backward_abductions



def bi_chainer():

    # load datasets
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--dataset", type=str, default="./data/proof_writer/test.json")
    arg_parse.add_argument("--model", type=str, default="llama")
    arg_parse.add_argument("--device", type=str, default="cuda")
    arg_parse.add_argument("--model_dir", type=str, default="./llama2-7b-chat")
    arg_parse.add_argument("--examples", type=str, default="./examples/proof_writer.txt")
    arg_parse.add_argument("--temp", type=float, default=0.1)
    arg_parse.add_argument("--max_new_tokens", type=float, default=384)
    arg_parse.add_argument("--max_reasoning_step", type=float, default=30)
    arg_parse.add_argument("--output_dir", type=str, default="./exp_results")

    args = arg_parse.parse_args()

    device = args.device


    with open(args.dataset, "r", encoding="utf8") as fp:
        proofwriter_data = json.load(fp)

    with open(args.examples, "r", encoding="utf8") as fp:
        fact_check_examples, forward_rule_selection_examples, logical_deduction_examples, confusion_check_examples = json.load(fp)

    # load model
    model, tokenizer = load_model(args)

    for idx, data in enumerate(proofwriter_data):
        data = proofwriter_data[idx]

        hypothesis, facts, rules = data_process(data)

        step = 0
        while step < args.max_reasoning_step:

            condition_consequences = do_condition_consequence_identify(hypothesis, facts, rules, tokenizer, model, device)

            identified_facts = do_fact_identify(hypothesis, facts, tokenizer, model, device)
            step += 1

            if len(identified_facts) > 0:
                # do forward chaining
                direction_flag = "forward"
            else:
                direction_flag = "backward"

            if direction_flag == "forward":
                step, fact_check_result, direction_flag, facts, identified_facts = one_forward_chaining(step, hypothesis, facts, rules, identified_facts, fact_check_examples, forward_rule_selection_examples, logical_deduction_examples, confusion_check_examples, tokenizer, model, device)

                if fact_check_result != "Unknown":
                    print("The asnwer is:", fact_check_result)
                    break

            if direction_flag == "backward":
                step, fact_check_result, direction_flag, facts, identified_facts = one_backward_chaining(step, hypothesis, facts, rules, identified_facts, fact_check_examples, forward_rule_selection_examples, logical_deduction_examples, confusion_check_examples, tokenizer, model, device)

                if fact_check_result != "Unknown":
                    print("The asnwer is:", fact_check_result)
                    break








if __name__ == "__main__": 
    seed_list = [1006, 2048, 1093]
    for seed in seed_list:
        bi_chainer()








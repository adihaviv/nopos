import os
import argparse
from tqdm import tqdm
import shutil
import json


def get_pe_from_name(path):
    if "-npe-" in path:
        return "npe"
    elif "-alibi-" in path:
        return "alibi"
    elif "-learned-" in path:
        return "learned"
    elif "-sinusoidal-" in path:
        return "sinusoidal"
    else:
        raise Exception("file name should include npe, alibi, learned or sinusoidal.\n{}".format(path))


def get_parameter_value(parameters_str, key, is_bool=False):
    key = "'"+key+"':"
    a = parameters_str[parameters_str.find(key) + len(key):]
    end_idx = a.find(",") if not is_bool else a.find("}")
    return a[:end_idx].strip()


def get_experiment_params(lines):
    for line in lines:
        first_line_prefix = line.find("fairseq_cli.train |")
        if first_line_prefix == -1:
            continue
        parameters_str = line[first_line_prefix + len("fairseq_cli.train |"):]
        return parameters_str
    raise Exception("Could not find experiment params")


def get_validation_value(key, validation_parts):
    for part in validation_parts:
        if key in part:
            return part.replace(key, "").strip()
    return ("{} not found in validation stats:{}.".format(key, validation_parts))


def get_last_validation_data(lines):
    last_validation_line_ct = 1
    while "valid on 'valid' subset |" not in lines[-last_validation_line_ct] and \
            last_validation_line_ct < len(lines):
        last_validation_line_ct += 1
    validation_line = lines[-last_validation_line_ct]
    return validation_line


def get_best_validation_data(metric, lines):
    best_mad = 10000.0
    best_validation_line = lines[0]
    for line in lines:
        if "valid on 'valid' subset |" in line and metric in line:
            mad_val = get_validation_value(metric, line.split("|"))
            if float(mad_val) < best_mad:
                best_mad = float(mad_val)
                best_validation_line = line

    return best_validation_line


def get_stats(path):
    res_files = [os.path.join(path, f) for f in tqdm(os.listdir(path)) if f.endswith(".out")]

    with open(os.path.join(path, "stats_last.txt"), "w") as out_fl:
        out_fl.write("Experiment\tLayer\tLR\tProbe Model\tMAD\tAccuracy\tPPL\n")
        for file in tqdm(res_files):
            pe = get_pe_from_name(os.path.basename(file))
            with open(file, "r", encoding="latin-1") as in_f:
                lines = in_f.readlines()
                parameters_str = get_experiment_params(lines)
                #parameters = json.load(parameters_str)
                layer_id = get_parameter_value(parameters_str, "probe_layer_idx")
                lr = get_parameter_value(parameters_str, "lr")[2:-1]
                non_linear = "linear" if get_parameter_value(parameters_str, "non_linear_probe", is_bool=True) == 'False' else "non-linear"

                last_validation_stats = get_last_validation_data(lines).split("|")
                ppl = get_validation_value("ppl", last_validation_stats)
                acc = get_validation_value("accuracy", last_validation_stats)
                mad = get_validation_value("mad", last_validation_stats)
                out_fl.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(pe, layer_id, lr, non_linear, mad, acc, ppl))

    if args.best_metric:
        with open(os.path.join(path, "stats_best.txt"), "w") as out_fb:
            out_fb.write("Experiment\tLayer\tMAD\tAccuracy\tPPL\n")
            for file in tqdm(res_files):
                pe = get_pe_from_name(os.path.basename(file))
                with open(file, "r", encoding="latin-1") as in_f:
                    lines = in_f.readlines()
                    parameters_str = get_experiment_params(lines)
                    #parameters = json.load(parameters_str)
                    layer_id = get_parameter_value(parameters_str, "probe_layer_idx")
                    lr = get_parameter_value(parameters_str, "lr")[2:-1]
                    non_linear = "non-linear" if bool(get_parameter_value(parameters_str, "non_linear_probe", is_bool=True)) else "linear"

                    best_validation_stats = get_best_validation_data(args.best_metric, lines).split("|")
                    ppl = get_validation_value("ppl", best_validation_stats)
                    acc = get_validation_value("accuracy", best_validation_stats)
                    mad = get_validation_value("mad", best_validation_stats)
                    out_fb.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(pe, layer_id, lr, non_linear, mad, acc, ppl))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        default=r"ddp-pile-results",
                        type=str,
                        required=False)

    parser.add_argument("--best_metric",
                        default=None,
                        type=str,
                        required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    get_stats(args.dir)
    print("Done")
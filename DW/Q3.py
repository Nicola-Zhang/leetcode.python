import xlrd
import numpy as np
import json


src_key_words = ["原文", "预测标签", "预测来源", "系统编号", "人工标注"]


def read_from_xlsx(file_path):
    books = xlrd.open_workbook(filename=file_path)
    sheets = books.sheets()[0]
    nrows, ncols = sheets.nrows, sheets.ncols

    title_name = sheets.row_values(0)

    samples = {}  # 系统编号->原文->算法来源->结果
    # text2sys, sys2text = {}, {}
    for i in range(1, nrows-1):
        sample = {}
        for k ,v in zip(title_name, sheets.row_values(i)):
            sample[k] = v

        sys_id = sample['系统编号']
        text = sample['原文']
        algo_id = sample['预测来源']
        man_result = 0 if sample['人工标注']=='对' else 1  # man_result is the index of array

        # print(f'原文：{text}, 预测来源：{algo_id}, 人工标注：{man_result}.')

        if sys_id not in samples:
            result = {algo_id: np.zeros(2)}
            samples[sys_id] = {text: result}
        else:
            if text not in samples[sys_id]:
                result = {algo_id: np.zeros(2)}
                samples[sys_id][text] = result
            else:
                if algo_id not in samples[sys_id][text]:
                    samples[sys_id][text][algo_id] = np.zeros(2)
        samples[sys_id][text][algo_id][man_result] += 1

    # print(samples)
    return samples


def count(samples, output_path):
    for sys_id, texts in samples.items():
        total_num, total_right_num = len(texts), 0
        algo1_num, algo1_right_num = 0, 0
        algo2_num, algo2_right_num = 0, 0

        for text, algos in texts.items():
            if_right = 0
            for algo, result in algos.items():
                if 'algo1' in algo:
                    algo1_num += np.sum(result)
                    algo1_right_num += result[0]
                    if_right += result[0]
                elif 'algo2' in algo:
                    algo2_num += np.sum(result)
                    algo2_right_num += result[0]
                    if_right += result[0]
                else:
                    raise ValueError
            if if_right > 0:
                total_right_num += 1

        print(f'{sys_id}: \n'
              f'total_num:{total_num}, total_recall:{total_right_num}'
              f'algo1_num:{algo1_num}, algo1_recall:{algo1_right_num}'
              f'algo2_num:{algo2_num}, algo2_recall:{algo2_right_num}.\n')

        final_result = {
            "系统编号": sys_id,
            "复审正确的个数": total_right_num,
            "algo1包含的预测结果数量": algo1_num,
            "algo-1预测正确的个数": algo1_right_num,
            "algo-2包含的预测结果数量": algo2_num,
            "algo-2预测正确的个数": algo2_right_num,
            "algo-1查全率": algo1_right_num/total_right_num,
            "algo-1查准率": algo1_right_num/algo1_num if algo1_num>0 else 1,
            "algo-2查全率": algo2_right_num/total_right_num,
            "algo-2查准率": algo2_right_num/algo2_num if algo2_num>0 else 1
        }
        with open(output_path, 'a+', encoding='utf-8') as fw:
            line = json.dumps(final_result, ensure_ascii=False)
            fw.write(line + "\n")


if __name__ == '__main__':
    input_path = "题目三.xlsx"
    output_path = "Q3.json"

    samples = read_from_xlsx(file_path=input_path)
    results = count(samples, output_path)
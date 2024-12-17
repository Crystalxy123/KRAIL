import gradio as gr
import pdb
import copy
import json
import base64
import time
import requests
import json
import numpy as np
import re
import gradio as gr
import random
import sys
import cv2
import os
import uuid
import openai
from openai import OpenAI
from PIL import Image
import io
import shutil
import markdown as md
import importlib
from analysis_process import *
from table_analysis import *
 
from neo4j import GraphDatabase, RoutingControl
import pdb
import ast
import argparse
import pandas as pd
import concurrent.futures
import itertools
from fuzzywuzzy import fuzz

       
URI = "your Neo4j url"
AUTH = ("name", "password")

base_tags = ['Scenario Familiarity','Information Availability and Reliability','Task Complexity']
weights_tags = ['Workplace Accessibility and Habitability','Workplace Visibility',\
                'Workplace Noise','Cold/Heat/Humidity']


def judge_object(object):
    if object == "PIF Attributes and Base HEPs":
        return  gr.update(choices=base_tags, visible=True)
    elif object == "PIF Attributes and Weights":
        return gr.update(choices=weights_tags, visible=True)
    else:
        pass       

def save_selection(result_label):
    return f"{result_label}"

def get_lib(driver, table,arr):
    records, _, _ = driver.execute_query(
        f"MATCH (n:{table}) "
        f"RETURN n.{arr}",
         database_="neo4j", routing_=RoutingControl.READ,
    )
    list_converse = [
        (value.strip() if value else '')  # 如果值存在则调用 strip，否则用空字符串
        for record in records  
        if (value := record.get(f'n.{arr}', None)) is not None  # 获取键值并确保其不为 None
    ]
    return list_converse


def generate(history,prompt,model,temperature):
    if model == "gpt-4o" or model == "claude-3-5-sonnet-20240620" or model == "gpt-4-turbo":
        outlines_new = call_models(prompt, history, model)
    else:
        outlines_new = call_step1(prompt, history, model, temperature)
    if not outlines_new:
        return False
    scen = re.findall(r'<(.*?)>', outlines_new, re.DOTALL)
    if not scen:
        pdb.set_trace()
        return "匹配失败"
    else:
        pass
    return scen[0]
    

def update_scen(prompt, data_source, model,temperature):
    prompt2 = prompt+"\n待分析的内容为："+data_source
    history = [{"role": "user", "content": "开始" }]
    result = generate(history,prompt2,model,temperature)
    return result


def similar(data_source,df,i):
    df = df.drop(i)
    df_concat = df.iloc[:, 6:].apply(lambda row: ''.join(row.astype(str)), axis=1)
    # 计算目标字符串和拼接后字符串的相似度
    similarities = df_concat.apply(lambda x: fuzz.ratio(x, data_source))
    # 找出最相似的行号
    best_match_row = similarities.idxmax()
    # 获取该行的前六列内容
    best_match_front_cols = df.iloc[best_match_row, :6]
    best_match_rest_cols = df.iloc[best_match_row, 6:].astype(str).str.cat()
    return best_match_front_cols,best_match_rest_cols

def update_base_HEPs(scen_task,scen_context,scen_cog,scen_time,table1,table2,relationship,data_source, model,temperature,df,i):
    best_match_front_cols,best_match_rest_cols = similar(data_source,df,i)
    learn_pif="已知对于任务"+best_match_rest_cols+"的PIF为"+best_match_front_cols.tolist()[0]
    learn_CFM="已知对于任务"+str(best_match_rest_cols)+"的CFM为"+str(best_match_front_cols.tolist()[1])
    learn_task="已知对于任务"+best_match_rest_cols+"的Task (and error measure)为"+best_match_front_cols.tolist()[3]
    learn_measure="已知对于任务"+best_match_rest_cols+"的PIF Measure为"+best_match_front_cols.tolist()[4]
    learn_other="已知对于任务"+str(best_match_rest_cols)+"的Other PIFs (and Uncertainty)为"+str(best_match_front_cols.tolist()[5])



    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        task_lib = list(set(get_lib(driver, table2,"Task_and_error_measure")))

        print("------1.task------")
        history_task = [{"role": "user", "content": "现在请你分析the tasks for which the human error rates were reported in the data source.along with the definition of the human errors measured for the tasks.如下是用于分析的参考信息，理解后请回复：“好的，我会参考这些信息来分析Task (and error measure)。”：" + "\n" + "## 知识库”：" + "\n" + "{" + "## task" + scen_task + "\n" + "## context" + scen_context +  "\n"+"## cognitive activities" + scen_cog+"\n"+"## time constraints"+scen_time+"\n"},
        {"role": "assistant", "content": "好的，我会参考这些信息来分析Task (and error measure)。"},
        {"role": "user", "content": 
        "请以如下的信息要求来撰写："  + f"请你从{task_lib}中选择最适合的三个，输出Task (and error measure),error measure在()里面，输出分析过程，最终结果输出在<>之间，不同的task用\',\'隔开，用英文回答我的问题，如{example_task}。"\
            +"请严格按照要求格式进行输出。" }]
        
        task_error_measure = generate(history_task,"",model,temperature)
        print(task_error_measure)

        print("------2.CFMs------")
        records, _, _ = driver.execute_query(
            f"MATCH (n:{table2}) WHERE n.Task_and_error_measure CONTAINS \"{task_error_measure}\""
            f"RETURN n.CFM",
            database_="neo4j", routing_=RoutingControl.READ,
        )
        CFMs_lib = [
            (value.strip() if value else '')  # 如果值存在则调用 strip，否则用空字符串
            for record in records  
            if (value := record.get(f'n.{"CFM"}', None)) is not None  # 获取键值并确保其不为 None
        ]

        history_CFMs = [{"role": "user", "content":'''现在请你分析CFMs.The CFMs are labeled as D, U, DM, E, and T for failure of Detection, Understanding, Decisionmaking, Action execution, and Interteam Coordination. Note that the task may have multiple applicable CFMs.
            1. D（Detection）：未能正确检测或识别所需信息（如报警、仪表读数）。
            2. U（Understanding）：未能正确理解或解释信息（如误解程序、错误解读系统状态）。
            3. DM（Decisionmaking）：未能做出正确决策（如选择了不合适的策略）。
            4. E（Action Execution）：未能正确执行所需动作（如按错按钮、操作延迟）。
            5. T（Interteam Coordination）：团队间沟通或协作失败（如信息传递错误或协调不畅）。
            If the task completion time is reported for an event in which applicable CFMs cannot be distinguished, then the output is “Unsp” for unspecified CFMs.
            如果过程涉及多个CFMs，请按照逻辑进行输出，&表示逻辑“与”，/表示逻辑“或”，假如过程同时存在D和U，则输出<D&U>
            Please determine the applicable CFMs based on the content, with the output as a list within < >, e.g., <Unsp>.如下是用于分析的参考信息，理解后请回复：“好的，我会参考这些信息来分析CFMs。”：''' + "\n" + "## 知识库”：" + "\n" + "{" + "## task" + scen_task + "\n" + "## context" + scen_context +  "\n"+"## cognitive activities" + scen_cog+"\n"+"## time constraints"+scen_time+"\n"+"## Task (and error measure)"+task_error_measure},
        {"role": "assistant", "content": "好的，我会参考这些信息来分析CFMs。"},
        {"role": "user", "content": 
        "请以如下的信息要求来撰写："  + f"请你从{CFMs_lib}中选择最适合三个，的输出CFMs,输出分析过程，最终结果输出在<>之间，不同的CFM用\',\'隔开，用英文回答我的问题，如{example_CFMs}。"\
            +"请严格按照要求格式进行输出。" }]
        cfms = generate(history_CFMs,"",model,temperature)
        print(cfms)

        print("-----3.relevent_pif-----")
        records, _, _ = driver.execute_query(
            f"MATCH (n:{table1})-[r:{relationship}]->(m:{table2}) WHERE (m.Task_and_error_measure CONTAINS \"{task_error_measure}\" AND m.CFM CONTAINS \"{cfms}\")"
            f"RETURN n.PIF_Attribute",
            database_="neo4j", routing_=RoutingControl.READ,
        )
        PIF_Attribute_lib = [
            (value.strip() if value else '')  # 如果值存在则调用 strip，否则用空字符串
            for record in records  
            if (value := record.get(f'n.{"PIF_Attribute"}', None)) is not None  # 获取键值并确保其不为 None
        ]
        history_PIF = [{"role": "user", "content":'''现在请你分析the base PIF attribute.如下是用于分析的参考信息，理解后请回复：“好的，我会参考这些信息来分析base PIF attribute。”：''' + "\n" + "## 知识库”：" + "\n" + "{" + "## task" + scen_task + "\n" + "## context" + scen_context +  "\n"+"## cognitive activities" + scen_cog+"\n"+"## time constraints"+scen_time+"\n"+"## Task (and error measure)"+task_error_measure+"\n"+"## CFMs"+cfms},
        {"role": "assistant", "content": "好的，我会参考这些信息来分析base PIF attribute。"},
        {"role": "user", "content": 
        "请以如下的信息要求来撰写："  + "请你从"+str(PIF_Attribute_lib)+f"中选择最适合的三个，输出PIF attribute,输出分析过程，最终结果输出在<>之间，不同的PIF attribute用\',\'隔开，用英文回答我的问题，如{example_PIF_Attri}。"+"请严格按照要求格式进行输出。" }]
        relevent_pif = generate(history_PIF,"",model,temperature)
        print(relevent_pif)

        print("-----4.PIF_measure------")
        records, _, _ = driver.execute_query(
            f"MATCH (n:{table1})-[r:{relationship}]->(m:{table2}) WHERE (m.Task_and_error_measure CONTAINS \"{task_error_measure}\" AND m.CFM CONTAINS \"{cfms}\" AND n.PIF_Attribute CONTAINS \"{relevent_pif}\")"
            f"RETURN m.PIF_Measure",
            database_="neo4j", routing_=RoutingControl.READ,
        )
        PIF_Measure_lib = [
            (value.strip() if value else '')  # 如果值存在则调用 strip，否则用空字符串
            for record in records  
            if (value := record.get(f'm.{"PIF_Measure"}', None)) is not None  # 获取键值并确保其不为 None
        ]
        history_PIF_Measure = [{"role": "user", "content":'''现在请你分析PIF attribute measure -The task-specific factor or variable used in the datasource under which the tasks were performed and human error rates were measured.如下是用于分析的参考信息，理解后请回复：“好的，我会参考这些信息来分析PIF attribute measure。”：''' + "\n" + "## 知识库”：" + "\n" + "{" + "## task" + scen_task + "\n" + "## context" + scen_context +  "\n"+"## cognitive activities" + scen_cog+"\n"+"## time constraints"+scen_time+"\n"+"## Task (and error measure)"+task_error_measure+"\n"+"## CFMs"+cfms+"## base PIF attribute"+relevent_pif},
        {"role": "assistant", "content": "好的，我会参考这些信息来分析PIF attribute measure。"},
        {"role": "user", "content": 
        "请以如下的信息要求来撰写："  + f"请你从{PIF_Measure_lib}中选择最适合的三个输出PIF attribute measure,输出分析过程，最终结果输出在<>之间，不同的PIF attribute measure用\',\'隔开，用英文回答我的问题，如{example_PIF_measure}。"\
            +"请严格按照要求格式进行输出。" }]
        PIF_measure = generate(history_PIF_Measure,"",model,temperature)
        print(PIF_measure)

        print("-----5.other_pif_uncertainties-----s")
        records, _, _ = driver.execute_query(
            f"MATCH (n:{table1})-[r:{relationship}]->(m:{table2}) WHERE (m.Task_and_error_measure CONTAINS \"{task_error_measure}\" AND m.CFM CONTAINS \"{cfms}\" AND n.PIF_Attribute CONTAINS \"{relevent_pif}\" AND m.PIF_Measure CONTAINS \"{PIF_measure}\")"
            f"RETURN m.Other_PIFs_and_Uncertainty",
            database_="neo4j", routing_=RoutingControl.READ,
        )
        Other_pif_lib = [
            (value.strip() if value else '')  # 如果值存在则调用 strip，否则用空字符串
            for record in records  
            if (value := record.get(f'm.{"Other_PIFs_and_Uncertainty"}', None)) is not None  # 获取键值并确保其不为 None
        ]
        history_other_PIF = [{"role": "user", "content":'''现在请你分析Other PIFs (and Uncertainty).Other PIFs that are also present in the tasks and uncertainties -In addition to the PIFs attribute that were under the study, the context of the tasks in a data sourcemay have other PIF attributes present during task performance; therefore, they would contribute to the reported error rates. It documents other PIF attributes thatwere present. In particular, it documents whether the tasks were performedunder time constraints. Information about the time availability is important to infer thebase HEPs from the reported human error data. lf the time available is inadequate, thena reported human error rate corresponds the probabilistic sum of the base HEPs and theerror probability due to inadequate time (P,). It also documents the uncertaintiesin the data source and in the mapping to the CFMs and PIF attributes. The uncertaintieswould affect how the reported error rates are to be integrated to inform base HEPS.
        There are uncertainties in the data source and in the mapping to IDHEAS-G CFMs. In particular, if the number of the times the task was performed is not sufficiently large, the reported error rate may not represent the lowest HEP.如下是用于分析的参考信息，理解后请回复：“好的，我会参考这些信息来分析Other PIFs (and Uncertainty)”：''' + "\n" + "## 知识库”：" + "\n" + "{" + "## task" + scen_task + "\n" + "## context" + scen_context +  "\n"+"## cognitive activities" + scen_cog+"\n"+"## time constraints"+scen_time+"\n"+"## Task (and error measure)"+task_error_measure+"\n"+"## CFMs"+cfms+"## base PIF attribute"+relevent_pif +"## PIF Measure"+PIF_measure},
        {"role": "assistant", "content": "好的，我会参考这些信息来分析Other PIFs (and Uncertainty)。"},
        {"role": "user", "content": 
        "请以如下的信息要求来撰写："  + f"请你从{Other_pif_lib}中选择最适合的三个输出Other PIFs (and Uncertainty),（）里面的内容是不确定性的内容，输出分析过程，最终结果输出在<>之间，不同的Other_PIFs_and_Uncertainty用\',\'隔开，用英文回答我的问题，如{example_other_PIF_uncertainties}。"\
            +"请严格按照要求格式进行输出。" }]
        other_pif_uncertainties = generate(history_other_PIF,"",model,temperature)
        print(other_pif_uncertainties)

    return task_error_measure,cfms,relevent_pif,PIF_measure,other_pif_uncertainties



def update_info(scen_task,scen_context,scen_cog,scen_time,object_input,result_label,data_source, model,temperature,df,i):
    if object_input == "PIF Attributes and Base HEPs":
        if result_label == "Scenario Familiarity":
            table1 = "Table_A1_1"
            table2 = "Table_A1_2"
            relationship = "RELATED_TO_A1" 
        elif result_label == "Information Availability and Reliability":
            table1 = "Table_A2_1"
            table2 = "Table_A2_2"
            relationship = "RELATED_TO_A2" 

        elif result_label == "Task Complexity":
            table1 = "Table_A3_1"
            table2 = "Table_A3_2"
            relationship = "RELATED_TO_A3" 
        else:
            pass
        task_error_measure,cfms,relevent_pif,PIF_measure,other_pif_uncertainties = update_base_HEPs(scen_task,scen_context,scen_cog,scen_time,table1,table2,relationship,data_source, model,temperature,df,i)

    elif object_input == "PIF Attributes and Weights":
        pass
    else:
        pass

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        task_lib_list = list(set(get_lib(driver, table2,"Task_and_error_measure")))
        CFMs_lib_list = list(set(get_lib(driver, table2,"CFM")))
        other_pif_lib_list = list(set(get_lib(driver, table2,"Other_PIFs_and_Uncertainty")))
        PIFs_measure_lib_list = list(set(get_lib(driver, table2,"PIF_Measure")))
        PIF_Attribute_lib_list = list(set(get_lib(driver, table1,"PIF_Attribute")))
    return task_error_measure,cfms,relevent_pif,other_pif_uncertainties,PIF_measure,table2,table1,relationship,task_lib_list,CFMs_lib_list,other_pif_lib_list,PIFs_measure_lib_list,PIF_Attribute_lib_list



def process(value_data_source,num,df,i):
    model = "claude-3-5-sonnet-20240620"
    temperature =0.9
    data_source = value_data_source
    result_labels=["Scenario Familiarity","Information Availability and Reliability","Task Complexity"]
    result_label_state=result_labels[2]
    object_input_state="PIF Attributes and Base HEPs"#PIF Attributes and Weights
    ## 分析过程
    prompt_analyze_task2 = prompt_analyze_task
    prompt_analyze_context2 = prompt_analyze_context
    prompt_analyze_cog2 =prompt_analyze_cog
    prompt_analyze_time2 = prompt_analyze_time

    scen_task = update_scen(prompt_analyze_task2,data_source,model,temperature)
    print("----------scen_task---------")
    print(scen_task)
    scen_context =update_scen(prompt_analyze_context2,data_source,model,temperature)
    print("----------scen_context---------")
    print(scen_context)
    scen_cog = update_scen(prompt_analyze_cog2,data_source,model,temperature)
    print("----------scen_cog---------")
    print(scen_cog)
    scen_time = update_scen(prompt_analyze_time2,data_source,model,temperature)
    print("----------scen_time---------")
    print(scen_time)
        
    task_error_measure,cfms,relevent_pif,other_pif_uncertainties,PIF_measure,table2,table1,relationship,task_lib_list,CFMs_lib_list,other_pif_lib_list,PIFs_measure_lib_list,PIF_Attribute_lib_list=update_info(scen_task,scen_context,scen_cog,scen_time,object_input_state,result_label_state,data_source,model,temperature,df,i)
    return task_error_measure,cfms,relevent_pif,other_pif_uncertainties,PIF_measure,table2,table1,relationship,task_lib_list,CFMs_lib_list,other_pif_lib_list,PIFs_measure_lib_list,PIF_Attribute_lib_list
       
def process_row(i, row,df_columns,output_dir,num,df):
    # 提取前六列的内容
    out_path = os.path.join(output_dir, f"data{i}.json")
    print("-----------out_path----------")
    print(out_path)
    if os.path.exists(out_path):
        print(f"Skipping {i}, file already exists: {out_path}")
        return None  # 文件已存在，跳过
    fixed_values = row[:6].values.tolist()
    
    # 将第七列及之后的所有内容拼接成一个字符串
    concatenated_values = ' '.join(str(row[col]) for col in df_columns[6:])
    
    # 获取处理结果
    result = process(concatenated_values,num,df,i)
    
    # 将前六列和处理结果合并为JSON格式
    output_row_json = {
        'i':i,
        'df_columns[6]':df_columns[6],
        'task_error_measure': result[0],
        'cfms': result[1],
        'relevent_pif': result[2],
        'other_pif_uncertainties': result[3],
        'PIF_measure': result[4],
        'table2': result[5],
        'table1': result[6],
        'relationship': result[7],
        'task_lib_list': result[8],
        'CFMs_lib_list': result[9],
        'other_pif_lib_list': result[10],
        'PIFs_measure_lib_list': result[11],
        'PIF_Attribute_lib_list': result[12],
        'fixed_values[0]':fixed_values[0],
        'fixed_values[1]':fixed_values[1],
        'fixed_values[2]':fixed_values[2],
        'fixed_values[3]':fixed_values[3],
        'fixed_values[4]':fixed_values[4],
        'fixed_values[5]':fixed_values[5]
    }

    # 保存到JSON文件
    with open(out_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_row_json, json_file, ensure_ascii=False, indent=4)
    
    # 返回前六列和处理结果，用于其他操作
    output_row = fixed_values + list(result)
    return output_row

# 创建 Gradio 界面
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate statistics of experts")
    parser.add_argument("--debug", type=str, default="False", help="debug or not")
    parser.add_argument("--excel_path", type=str, default="", help="debug or not")
    parser.add_argument("--parall", type=str, default="False", help="debug or not")
    parser.add_argument("--dir", type=str, default="table1", help="debug or not")
    parser.add_argument("--num", type=int, default=0, help="debug or not")
    parser.add_argument("--output", type=str, default="", help="debug or not")




    args = parser.parse_args()
    output_data = []
    df =  pd.read_excel(args.excel_path)
    fixed_columns = df.columns[:6]
    i=0
    output_dir=f"{args.output}/{args.dir}"

    if args.parall=="True":
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_row, 
                            range(len(df)), 
                            [df.iloc[i] for i in range(len(df))], 
                            [df.columns] * len(df), 
                            [output_dir] * len(df),
                            itertools.repeat(args.num, len(df)),
                            itertools.repeat(df, len(df))))  # 传递 df
        output_data.extend(results)
    else:
        for idx, row in df.iterrows():
            i=i+1
            out_path = os.path.join(output_dir, f"data{i}.json")
            if os.path.exists(out_path):
                print(f"Skipping {i}, file already exists: {out_path}")
                continue
    
            # 提取前六列的内容
            fixed_values = row[:6].values.tolist()
            # 将第七列及之后的所有内容拼接成一个字符串
            concatenated_values = ' '.join(str(row[col]) for col in df.columns[6:])
            # 处理第七列及之后的内容
            
            
            result = process(concatenated_values,args.num,df,i)
            
            # 将前六列和处理结果合并
            output_row = fixed_values + list(result)
            output_data.append(output_row)
            # 将前六列和处理结果合并
            output_row_json = ({
                'task_error_measure': result[0],
                'cfms': result[1],
                'relevent_pif': result[2],
                'other_pif_uncertainties': result[3],
                'PIF_measure': result[4],
                'table2': result[5],
                'table1': result[6],
                'relationship': result[7],
                'task_lib_list': result[8],
                'CFMs_lib_list': result[9],
                'other_pif_lib_list': result[10],
                'PIFs_measure_lib_list': result[11],
                'PIF_Attribute_lib_list': result[12]
            })
            # 保存到JSON文件
            with open(out_path, 'w', encoding='utf-8') as json_file:
                json.dump(output_row_json, json_file, ensure_ascii=False, indent=4)
            # 将结果添加到output_data中
            output_data.append(output_row)



    # 创建新的DataFrame
    output_columns = list(fixed_columns) + [
        'task_error_measure', 'cfms', 'relevent_pif', 'other_pif_uncertainties', 'PIF_measure', 'table2', 
        'table1', 'relationship', 'task_lib_list', 'CFMs_lib_list', 'other_pif_lib_list', 
        'PIFs_measure_lib_list', 'PIF_Attribute_lib_list'
    ]
    output_df = pd.DataFrame(output_data, columns=output_columns)

    # 保存到新的Excel文件
    output_df.to_excel('processed_output1.xlsx', index=False)
    

   
    

    # 设计实验---
    # human in the loop
    # 刚开始agent推理的不对，但是修正，你不对，error rate，修正知识链，然后让多次迭代
    # base，
    # 定性，
    # case study
    # 消融实验：没有agent，没有图， 
    # claude 4o
    # 



    # 图的结构增强
    # 


    # 训练一个垂类模型，把它套到整个框架，7B左右的模型
    


    # 

'''
Author: your name
Date: 2021-01-08 14:32:24
LastEditTime: 2021-01-11 09:17:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /opt/sample/1.py
'''

import numpy as np
import csv

# 取字符对应的索引表示该字符
def find_index(x,y):
    return [i for i in range(len(y)) if y[i]==x]   # Python列表解析,返回列表

def handleProtocol(inputs):
    protocol_list=['tcp','udp','icmp']
    if inputs[1] in protocol_list:
        return find_index(inputs[1], protocol_list)[0]

def handleService(inputs):
    service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                 'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                 'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                 'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                 'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                 'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                 'uucp','uucp_path','vmnet','whois','X11','Z39_50']
    if inputs[2] in service_list:
        return find_index(inputs[2],service_list)[0]

def handleFlag(inputs):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    if inputs[3] in flag_list:
        return find_index(inputs[3],flag_list)[0]

def handleLabel(inputs):
    global label_list  # 定义label_list为全局变量，用于存放39种攻击类型
    label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
                'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
                'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
                'spy.', 'rootkit.']
    # 在函数内部使用全局变量并修改它
    if inputs[41] in label_list:
        return find_index(inputs[41],label_list)[0]
    else:
        label_list.append(inputs[41])                # 如果发现出现新的攻击类型，将它添加到label_list
        return find_index(inputs[41],label_list)[0]

# 文件写入
data_numerization = open("kdd.correct.num1.txt", 'w', newline='')  # 新建文件用于存放数值化后的数据集
if __name__=='__main__':
    with open('corrected1','r') as data_original:                       # 打开原始数据集文件
        csv_reader = csv.reader(data_original)                               # 按行读取所有数据并返回由csv文件的每行组成的列表
        csv_writer = csv.writer(data_numerization, dialect='excel')          # 先传入文件句柄
        for row in csv_reader:                       # 循环读取数据
            temp_line=np.array(row)                  # 将列表list转换为ndarray数组。                  
            temp_line[1] = handleProtocol(row)       # 将源文件行中3种协议类型转换成数字标识
            temp_line[2] = handleService(row)        # 将源文件行中70种网络服务类型转换成数字标识
            temp_line[3] = handleFlag(row)           # 将源文件行中11种网络连接状态转换成数字标识
            temp_line[41] = handleLabel(row)         # 将源文件行中23种攻击类型转换成数字标识
            csv_writer.writerow(temp_line)           # 按行写入
        data_numerization.close()
        print('数值化done！')
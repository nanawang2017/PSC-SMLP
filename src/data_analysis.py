# coding=utf-8
"""
Step1：
对前期工作进行汇总以及修改 创建于2019/03/02
对处理过的（通过用户签到数量和POI被签到数量筛选掉一部分用户和POI）Gowalla 和yelp数据进行处理
分别得到user_poi.pkl 和 user_poi.pkl
两个文件的用户完全相交 之前用户不相交的错误在于
自己在生成user_poi的时候没有将字符转化为int类型 导致 user_poi中的key 为字符型
user_friend 字典中的key是int 类型
根据范围统计数据
"""
import os
import pickle


def transfer_data_to_dict(file_path):
    data_dict = {}
    with open(file_path, 'r')as f:
        lines = f.readlines()
        for line in lines:
            key, value = int(line.split('\t')[0]), int(line.split('\t')[1])
            if key not in data_dict.keys():
                data_dict[key] = []
            data_dict[key].append(value)
    return data_dict


if __name__ == '__main__':
    user_friend = {}
    user_poi = {}
    poi_user={}
    project_path = os.path.dirname(os.getcwd())
    data_name='Gowalla'
    checkins_file = project_path + '/data/'+data_name+'/An-Experiment/'+data_name+'_check_ins.txt'
    social_file = project_path + '/data/'+data_name+'/An-Experiment/'+data_name+'_social_relations.txt'
    checkins_dict_path = project_path + '/data/'+data_name+'/inter/'+'user_poi.pkl'
    social_dict_path = project_path + '/data/'+data_name+'/inter/'+'user_friend.pkl'
    writeToFile = False
    try:
        f = open(checkins_dict_path, 'rb')
        f.close()
    except IOError:
        writeToFile = True

    if writeToFile:
        # 获得用户签到关系 以user_poi 字典形式存储 user：[poi list]
        user_poi = transfer_data_to_dict(checkins_file)
        print(len(user_poi.keys()))
        # 获得社交关系
        user_friend = transfer_data_to_dict(social_file)
        print(len(user_friend.keys()))
        with open(checkins_dict_path, 'wb+')as f:
            pickle.dump(user_poi, f)
        with open(social_dict_path, 'wb+')as f:
            pickle.dump(user_friend, f)
        print(len(set(user_poi.keys()).intersection(user_friend.keys())))
    else:
        print('The first dealing of the data is finished! ')
        user_lower,user_lupper=21,31
        poi_lower,poi_upper=21,31

        with open(checkins_dict_path, 'rb')as f:
            user_poi=pickle.load(f)
            print(len(list(user_poi.keys())))
        for user in list(user_poi.keys()):
            poi_array=user_poi[user]
            for poi in poi_array:
                if poi not in poi_user.keys():
                    poi_user[poi]=[]
                poi_user[poi].append(user)
        # 统计用户签到次数
        user_num = 0
        for user in list(user_poi.keys()):
            if user_lower <= len(user_poi[user]) < user_lupper:
                user_num+=1
        print('用户签到数量在{}和{}之间的数量为：{}'.format(user_lower,user_lupper,user_num))
        poi_num=0
        for poi in list(poi_user.keys()):
            if poi_lower <= len(poi_user[poi]) < poi_upper:
                poi_num+=1
        print('POI被签到数量在{}和{}之间的数量为：{}'.format(poi_lower,poi_upper,poi_num))


        # with open(checkins_dict_path, 'rb')as f:
        #     user_poi = pickle.load(f)
        #     print('Filtering users and spots')
        #     for user in list(user_poi.keys()):
        #         if len(user_poi[user]) < 15:
        #             del user_poi[user]
            # print(len(user_poi.keys()))
            # 1278

        # with open('/Users/wangnana/学习/小论文/cssc-eMLP/data/Gowalla/user_poi.pkl', 'rb')as f:
        #     user_friend = pickle.load(f)

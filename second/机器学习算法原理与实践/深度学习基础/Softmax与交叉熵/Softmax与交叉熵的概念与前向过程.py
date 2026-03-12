'''def split_string(s, delimiter):
    result = []
    del_len = len(delimiter)
    start = 0
    s_len = len(s)
    while start < s_len:
        # 查找分隔符的位置
        pos = s.find(delimiter, start)
        if pos == -1:
            # 没找到分隔符，添加剩余部分
            result.append(s[start:])
            break
        # 添加分隔符前的部分
        result.append(s[start:pos])
        # 跳过分隔符
        start = pos + del_len
    return result
def join_string(lst, delimiter):
    return delimiter.join(lst)
# 测试拆分
s = "ab&&2"
delimiter = "&&"
split_result = split_string(s, delimiter)
print(split_result)  # 输出：['ab', '2']

# 测试组合
lst = ['ab', '2']
join_result = join_string(lst, delimiter)
print(join_result)  # 输出：ab&&2'''

























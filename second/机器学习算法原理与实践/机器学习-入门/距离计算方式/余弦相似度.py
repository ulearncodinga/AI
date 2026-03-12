import math

x = [1,2]
y = [4,6]

def cosine_similarity(x,y):
    numerator = sum(a * b for a,b in zip(x,y))
    denominator = math.sqrt(sum(a ** 2 for a in x)) * math.sqrt(sum(b ** 2 for b in y))

    return numerator / denominator

print("余弦相似度:",cosine_similarity(x,y))
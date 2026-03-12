import math

x = [1,2]
y = [4,6]

def haversine_distance(lat1,lon1,lat2,lon2):
    R = 6371.0 #地球半径

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

#得到经纬度的差值
    dlon = lon2 - lon1
    dlat = lat2 - lat1


    d = 2 * R * math.asin(
       math.sqrt(
           math.sin(dlat / 2) ** 2
           + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
       )
    )

    return d

lat1,lon1 = 52.2296756,21.0122287 #华沙的经纬度
lat2,lon2 = 51.5073509,-0.1277583 #伦敦的经纬度

print("半正矢距离:",haversine_distance(lat1,lon1,lat2,lon2))


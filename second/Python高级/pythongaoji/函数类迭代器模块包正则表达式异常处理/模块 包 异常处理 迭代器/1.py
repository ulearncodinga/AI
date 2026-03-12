class Car():

    def __init__(self,name,color,type):
        self.name = name
        self.color = color
        self.type = type
    def move(self):
        print(f"这个车的颜色为{self.color},型号为{self.type}")


car = Car('bmw')

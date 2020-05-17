# class Car():
#     def __init__(self,make,model,year):
#         self.make = make
#         self.model = model
#         self.year = year
#         self.odometer_reading = 0
    
#     def get_descriptive(self):
#         long_name =str(self.year)+ ' '+ self.make + ' '+self.model
#         return long_name.title()

#     def read_odometer(self):
#         print("This car has "+str(self.odometer_reading) + " miles on it.")
    
#     def update_odometer(self,mileage):
#         self.odometer_reading = mileage

# class ElectricCar(Car):
#     def __init__(self,make,model,year):
#         super().__init__(make,model,year)

# my_car = Car('Audi','a4',2016)
# print(my_car.get_descriptive())
# my_car.update_odometer(24)
# my_car.read_odometer()

# my_tesla = ElectricCar('tesla','model s',2016)
# print(my_tesla.get_descriptive())

# with open('pi.txt') as file_object:
    # contents = file_object.read()
    # print(contents.rstrip())
    # for line in file_object:
    #     print(line.rstrip())
    # lines = file_object.readlines()

# pi_string = ''
# for line in lines:
#     pi_string += line.strip()

# print(pi_string)
# try:
#     print(5/0)
# except ZeroDivisionError:
#     print("ruozhi zhe dou neng cuo")

import json

numbers = [2,3,4,5,6,7,8,9]
filename = 'numbers.json'
with open(filename,'w') as fobj:
    json.dump(numbers,fobj)
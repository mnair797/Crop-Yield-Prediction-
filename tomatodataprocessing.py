def toyear(date):
  return date[len(date) - 4 :]

import pandas as pd
tomato_harvests = pd.read_csv('tomato_harvests_all.csv')
tomato_harvests = tomato_harvests.drop('tomato_hh_red_fresh_yield_Mg/ha', axis=1)
tomato_harvests = tomato_harvests.drop('tomato_hh_green_fresh_yield_Mg/ha', axis=1)
tomato_harvests = tomato_harvests.drop('tomato_hh_vine_dry_yield_Mg/ha', axis=1)
tomato_harvests = tomato_harvests.drop('tomato_mh_fresh_yield_Mg/ha', axis=1)
tomato_harvests = tomato_harvests.drop('tomato_fruit_moisture_%', axis=1)
tomato_harvests = tomato_harvests.drop('tomato_vine_moisture_%', axis=1)
tomato_harvests = tomato_harvests.drop('tomato_hh_vine_fresh_yield_Mg/ha', axis=1)

print ()
print ()
print ()

import datetime
weather = pd.read_csv('weather_data_davis3.csv')
#previous months
#march, april, may, june, july, august

#PRECIPITATION
precip = [0]*len(tomato_harvests)

#ETo
eto = [0]*len(tomato_harvests)


#SOLAR RADIATION
solrad = [0]*len(tomato_harvests)


#MAXIMUM AIR TEMPERATURE
maxairtemp = [0]*len(tomato_harvests)

#MINIMUM AIR TEMPERATURE
minairtemp = [0]*len(tomato_harvests)


#AVERAGE AIR TEMPERATURE
avgtemp = [0]*len(tomato_harvests)


#AVERAGE RELATIVE HUMIDITY
avgrelhum = [0]*len(tomato_harvests)


#DEW POINT
dew = [0]*len(tomato_harvests)


#AVERAGE WIND SPEED
avgwindspd = [0]*len(tomato_harvests)


#AVERAGE SOIL TEMPERATURE
avgsoiltemp = [0]*len(tomato_harvests)


#HEAT INDEX
heat_index_0 = [0]*len(tomato_harvests)
heat_index_1 = [0]*len(tomato_harvests)
heat_index_2 = [0]*len(tomato_harvests)
heat_index_3 = [0]*len(tomato_harvests)
heat_index_4 = [0]*len(tomato_harvests)

#DROGUHTS
droughts = [0]*len(tomato_harvests)


for ind in tomato_harvests.index:
  datee=datetime.datetime.strptime(tomato_harvests['date'][ind], "%m/%d/%Y")
  c0=0
  c1=0
  c2=0
  c3=0
  c4=0
  for w_ind in weather.index:
    datew=datetime.datetime.strptime(weather['Date'][w_ind], "%m/%d/%Y")  
    if datew.year==datee.year and datew.month >=3 and datew.month <=8:
      t = weather['Max Air Temp (F)'][w_ind]
      h = weather['Avg Rel Hum (%)'][w_ind]
      index=0
      if t>=80 and t<82:
        index=1
      elif t>=82 and t<84:
        if h<90:
          index=1
        else:
          index=2
      elif t>=84 and t<86:
        if h<75:
          index=1
        else:
          index=2
      elif t>=86 and t<88:
        if h>90:
          index=3
        elif h<=90 and h>=60:
          index=2
        else:
          index=1
      elif t>=88 and t<90:
        if h>75:
          index=3
        elif h<=75 and h>=50:
          index=2
        else:
          index=1
      elif t>=90 and t<92:
        if h>95:
          index=4
        elif h<=95 and h>=70:
          index=3
        else:
          index=2
      elif t>=92 and t<94:
        if h>85:
          index=4
        elif h<=85 and h>=60:
          index=3
        else:
          index=2
      elif t>=94 and t<96:
        if h>80:
          index=4
        elif h<=80 and h>=55:
          index=3
        else:
          index=2
      elif t>=96 and t<98:
        if h>70:
          index=4
        elif h<=70 and h>=45:
          index=3
        else:
          index=2
      elif t>=98 and t<100:
        if h<65:
          index=3
        else:
          index=4
      elif t>=100 and t<102:
        if h<60:
          index=3
        else:
          index=4
      elif t>=102 and t<104:
        if h<55:
          index=3
        else:
          index=4
      elif t>=104 and t<106:
        if h<50:
          index=3
        else:
          index=4
      elif t>=106 and t<108:
        if h<45:
          index=3
        else:
          index=4
      elif t>=108:
        index=4

      if index==0:
        c0+=1
      elif index==1:
        c1+=1
      elif index==2:
        c2+=1
      elif index==3:
        c3+=1
      elif index==4:
        c4+=1

  heat_index_0[ind]=c0
  heat_index_1[ind]=c1
  heat_index_2[ind]=c2
  heat_index_3[ind]=c3
  heat_index_4[ind]=c4

  if datee.year==2007 or datee.year==2008 or datee.year==2009 or datee.year==2013 or datee.year==2012 or datee.year==2011 or datee.year==2014:
    droughts[ind]=1
  else:
    droughts[ind]=0



tomato_harvests['heat_index_0']=heat_index_0
tomato_harvests['heat_index_1']=heat_index_1
tomato_harvests['heat_index_2']=heat_index_2
tomato_harvests['heat_index_3']=heat_index_3
tomato_harvests['heat_index_4']=heat_index_4
tomato_harvests['drought_presensce']=droughts


    


for ind in tomato_harvests.index:
  datee=datetime.datetime.strptime(tomato_harvests['date'][ind], "%m/%d/%Y")
  for w_ind in weather.index:
    datew=datetime.datetime.strptime(weather['Date'][w_ind], "%m/%d/%Y")   
    if datew.year==datee.year and datew.month>=3 and datew.month<=8:
      
      print ("HELLLOOO")
      precip[ind]+=weather['Precip (in)'][w_ind]
      eto[ind]+=weather['ETo (in)'][w_ind]
      solrad[ind]+=weather['Sol Rad (Ly/day)'][w_ind]
      maxairtemp[ind]+=weather['Max Air Temp (F)'][w_ind]
      avgrelhum[ind]+=weather['Avg Rel Hum (%)'][w_ind]  
      minairtemp[ind]+=weather['Min Air Temp (F)'][w_ind]
      avgtemp[ind]+=weather['Avg Air Temp (F)'][w_ind]
      dew[ind]+=weather['Dew Point (F)'][w_ind]
      avgwindspd[ind]+=weather['Avg Wind Speed (mph)'][w_ind]
      avgsoiltemp[ind]+=weather['Avg Soil Temp (F)'][w_ind]
  
  


tomato_harvests['precipitation']=precip
tomato_harvests['ETo']=eto
tomato_harvests['solrad']=solrad
tomato_harvests['maxairtemp']=maxairtemp
tomato_harvests['avgrelhum']=avgrelhum
tomato_harvests['minairtemp']=minairtemp
tomato_harvests['avgtemp']=avgtemp
tomato_harvests['avgtemp']=avgtemp
tomato_harvests['dew']=dew
tomato_harvests['avgwindspd']=avgwindspd
tomato_harvests['avgsoiltemp']=avgsoiltemp

print (tomato_harvests['maxairtemp'])


print (tomato_harvests.head())


for ind in tomato_harvests.index:
  tomato_harvests['date'][ind] = toyear(tomato_harvests['date'][ind])

pesticides = pd.read_csv('operations_pesticides.csv')
pesticides = pesticides.drop('method_application', axis=1)
pesticides = pesticides.drop('number_of_passes', axis=1)
pesticides = pesticides.drop('additional_description', axis=1)
pesticides = pesticides.drop('EPA_registration_number', axis=1)

for ind in pesticides.index:
  
  pesticides['date'][ind] = toyear(pesticides['date'][ind])


  if (pesticides['material_units'][ind]=='gallons/acre'):
    pesticides['material_quantity'][ind]=pesticides['material_quantity'][ind]*128
  if (pesticides['material_units'][ind]=='pints/acre'):
    pesticides['material_quantity'][ind]=pesticides['material_quantity'][ind]*16
  if (pesticides['material_units'][ind]=='pounds/acre'):
    pesticides['material_quantity'][ind]=pesticides['material_quantity'][ind]*16
  if (pesticides['material_units'][ind]=='quarts/acre'):
    pesticides['material_quantity'][ind]=pesticides['material_quantity'][ind]*32
  
    
   
pesticides = pesticides[pesticides.crop_applied == 'Tomato']

pestname=[-1]*len(tomato_harvests)
pestquantity = [-1]*len(tomato_harvests)

count=0
for ind in tomato_harvests.index:
  date = tomato_harvests['date'][ind]
  system_name = tomato_harvests['system_name'][ind]
  plot = tomato_harvests['plot'][ind]
  plot = plot[0:3]
  for pestind in pesticides.index:
    if date == pesticides['date'][pestind] and system_name==pesticides['system_name'][pestind] and plot==pesticides['plot'][pestind]:
      pestname[count] = pesticides['material_applied'][pestind]
      pestquantity[count] = pesticides['material_quantity'][pestind]
      break
  count+=1

tomato_harvests['pesticide_material_applied']=pestname
tomato_harvests['pesticide_material_quantity']=pestquantity


fertilizers = pd.read_csv('operations_fertilizers.csv')
fertilizers = fertilizers.drop('fertilizer_brand', axis=1)
fertilizers = fertilizers.drop('phosphorus_applied_kg/ha', axis=1)
fertilizers = fertilizers.drop('potassium_applied_kg/ha', axis=1)
fertilizers = fertilizers.drop('sulfur_applied_kg/ha', axis=1)

fertname=[-1]*len(tomato_harvests)
fertquantity = [-1]*len(tomato_harvests)

fertilizers = fertilizers[fertilizers.crop_fertilized == 'Tomato']
print (fertilizers.head())

for ind in fertilizers.index:
  fertilizers['date'][ind] = toyear(fertilizers['date'][ind])

count=0
for ind in tomato_harvests.index:
  date = tomato_harvests['date'][ind]
  system_name = tomato_harvests['system_name'][ind]
  plot = tomato_harvests['plot'][ind]
  plot = plot[0:3]
  for fertind in fertilizers.index:
    if date == fertilizers['date'][fertind] and system_name==fertilizers['system_name'][fertind] and plot==fertilizers['plot'][fertind]:
      fertname[count] = fertilizers['material_applied'][fertind]
      fertquantity[count] = fertilizers['total_applied_material_kg/ha'][fertind]
      break
  count+=1

tomato_harvests['fertilizer_material_applied']=fertname
tomato_harvests['fertilizer_material_quantity']=fertquantity

planting = pd.read_csv('operations_planting.csv')

planting = planting.drop('operation_type', axis=1)
planting = planting.drop('Tractor', axis=1)
planting = planting.drop('Equipment', axis=1)
planting = planting.drop('number_of_passes', axis=1)
planting = planting.drop('Additional_description', axis=1)
planting = planting.drop('Material_planted', axis=1)
planting = planting.drop('material_units', axis=1)

plantingquantity=[-1]*len(tomato_harvests)

planting = planting[planting.Crop_planted == 'Tomato']
print (planting.head())

for ind in planting.index:
  planting['Date'][ind] = toyear(planting['Date'][ind])

count=0
for ind in tomato_harvests.index:
  date = tomato_harvests['date'][ind]
  system_name = tomato_harvests['system_name'][ind]
  plot = tomato_harvests['plot'][ind]
  plot = plot[0:3]
  for plantind in planting.index:
    if date == planting['Date'][plantind] and system_name==planting['system_name'][plantind] and plot==planting['plot'][plantind]:
      plantingquantity[count] = planting['material_quantity'][plantind]
      break
  count+=1

tomato_harvests['planting_quantity']=plantingquantity

      


tomato_harvests = tomato_harvests[tomato_harvests.planting_quantity != -1]
tomato_harvests = tomato_harvests[tomato_harvests.pesticide_material_applied != -1]
tomato_harvests = tomato_harvests[tomato_harvests.pesticide_material_quantity != -1]
tomato_harvests = tomato_harvests[tomato_harvests.fertilizer_material_applied != -1]
tomato_harvests = tomato_harvests[tomato_harvests.fertilizer_material_quantity != -1]


tomato_harvests = tomato_harvests.drop('plot', axis=1)
tomato_harvests = tomato_harvests.drop('date', axis=1)


tomato_harvests.to_csv('cropyieldI.csv')


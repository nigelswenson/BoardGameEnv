import json

listthing=[]
for i in range(6):
    for j in range(6):
        listthing.append({'pattern':i, 'color':j})
with open('CalicoTileset.json', 'w+') as file:
    json.dump(listthing,file)


import csv
import os

rowData = []
path = os.getcwd()
print(path)
with open(path + '/dataset/MSR Video Description Corpus.csv', 'r', encoding='utf8') as file:
    reader = csv.reader(file)
    for row in reader:
        tempRow = []
        if len(row) > 0 and row[6] == 'English':
            tempRow.append(row[0]+'_'+row[1]+'_'+row[2]+'.avi')
            tempRow.append(row[7])
        if len(tempRow) == 2:
            rowData.append(tempRow)

# print(rowData)

fields = ['VideoID', 'Description']

# writing to csv file
with open(path+'/dataset/MSVD_description_cfile.csv', 'w', encoding="utf8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rowData)
print("done")

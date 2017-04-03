import csv

with open('reviews_data.csv','wb+') as write_file:
    csv_file = csv.writer(write_file)
    csv_file.writerow(['review_text','useful'])
    with open('reviews.csv') as csv_in:
        review_reader = csv.reader(csv_in, delimiter=',')
        next(review_reader)
        for r in review_reader:
            if int(r[1]) > 0:
                csv_file.writerow([r[0],1])
            else:
                csv_file.writerow([r[0],0])

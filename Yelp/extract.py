import json
import csv

with open('reviews.csv','wb+') as write_file:
    csv_file = csv.writer(write_file)
    csv_file.writerow(['review_text','useful','funny','cool'])
    with open('./yelp_dataset/yelp_academic_dataset_review.json') as data_file:
        for l in data_file:
            l_content = json.loads(l)
            csv_file.writerow([l_content['text'].encode('utf-8').strip(),
                               l_content['useful'],
                               l_content['funny'],
                               l_content['cool']])

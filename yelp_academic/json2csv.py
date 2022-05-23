import pandas as pd

# with open('json_data/yelp_academic_dataset_review.json', encoding='utf-8') as json_file:
#     df = pd.read_json(json_file, lines=True)
#
# df.to_csv('csv_data/yelp_academic_dataset_review.csv', encoding='utf-8', index=False)


# with open('json_data/yelp_academic_dataset_business.json', encoding='utf-8') as json_file:
#     df = pd.read_json(json_file, lines=True)
#
# df.to_csv('csv_data/yelp_academic_dataset_business.csv', encoding='utf-8', index=False)

with open('json_data/yelp_academic_dataset_user.json.json', encoding='utf-8') as json_file:
    df = pd.read_json(json_file, lines=True)

df.to_csv('csv_data/yelp_academic_dataset_user.csv', encoding='utf-8', index=False)

print('done')

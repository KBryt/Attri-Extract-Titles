``` The data is extracted from google MAVE data using the labels. Please visit https://github.com/google-research-datasets/MAVE```
  import json
with open('/content/drive/MyDrive/mave_positives.jsonl', 'r') as json_file:
      json_list = list(json_file)

      i = 0
      df = {}
      for json_str in json_list:
        json_dict = json.loads(json_str)
        df[i] = json_dict
        i += 1

        listDict= []
        for k, v in df.items():
          newDict = {}
          #newDict['id'] = v['id']
          #newDict['category'] = v['category']  
          for subdict in v['paragraphs']:
            if subdict['source']== 'title':
              newDict['title'] = subdict['text']
            # elif subdict['source']== 'description':
            #   newDict['description'] = subdict['text']
          for x in v['attributes']:
            newDict['attribute'] = x['key']
            newDict['value'] = x['evidences'][0]['value']

        listDict.append(newDict)

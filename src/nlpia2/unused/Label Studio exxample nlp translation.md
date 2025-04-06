Label Studio is an open source data labeling tool for labeling and exploring multiple types of data. You can perform different types of labeling with many data formats.

You can also integrate Label Studio with machine learning models to supply predictions for labels (pre-labels), or perform continuous active learning. See Set up machine learning with your labeling process.

Label Studio is also available in Enterprise and Cloud editions with additional features. See Label Studio features for more.


```json
[{
  # "data" must contain the "my_text" field defined in the text labeling config as the value and can optionally include other fields
  "data": {
    "my_text": "Opossums are great",
    "ref_id": 456,
    "meta_info": {
      "timestamp": "2020-03-09 18:15:28.212882",
      "location": "North Pole"
    } 
  },

  # annotations are not required and are the list of annotation results matching the labeling config schema
  "annotations": [{
    "result": [{
      "from_name": "sentiment_class",
      "to_name": "message",
      "type": "choices",
      "value": {
        "choices": ["Positive"]
      }
    }]
  }],

  # "predictions" are pretty similar to "annotations" 
  # except that they also include some ML-related fields like a prediction "score"
  "predictions": [{
    "result": [{
      "from_name": "sentiment_class",
      "to_name": "message",
      "type": "choices",
      "value": {
        "choices": ["Neutral"]
      }
    }],
  # score is used for active learning sampling mode
    "score": 0.95
  }]
}]
```

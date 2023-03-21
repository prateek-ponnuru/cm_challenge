CM_Challenge
================

The challenge consists of the following components -

- ### api
  1. GET /health/  
     To check the api service is running.
  2. POST /models/  
     To instantiate and save a model to database.
  3. GET /models/<int:model_id>/  
     To query a model from database.
  4. POST /models/<int:model_id>/train/  
     To incrementally train a model saved in the database.
  5. GET /models/<int:model_id>/predict/?x=<str:base64(x)>  
     To predict using a saved model.
  6. GET /models/  
     To get all saved models and calculate a score (normalized rank) on each class of models.
  7. GET /models/groups/  
     To get all models grouped by the number of incremental train steps performed.
- ### client  
     A Simple python script that uses pytest to test all the apis are working as expected.
- ### backend  
     A mysql service that is used by the api to save data persistently.
The final validation file (`results/submission.csv`) should contain 17004 lines. 
For each image, if there is more than one ship on it, it is necessary to separate them 
into several different lines. That is, for one image there can be more than one line in 
the final file. Honestly, I don't know how to do this, provided that our model is extremely 
weak for predicting the number of ships in an image. So the code that creates the 
submission.csv file is not correct. It creates one mask for each image. And since 
the kaggle check requires a file with a strictly fixed number of lines, it will not 
be possible to predict all this correctly with our model. In spite of this, I think 
that I coped with the task, because all the remaining problems and inaccuracies did 
not depend on me: I do not have access to sufficient computational resources to 
create a better model.

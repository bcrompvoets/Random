# NGC 3324 Notes

## Adding in 2MASS trained classifier
* Used IRAC only data to classify. 2MASS *appears* to perform well for the validation set, but as the validation set mostly includes the training set, this is not the best indicator. 

New training plan:
* Use 250 objects of each class as training sample. There are 376 EG, 1130 YSOs, and ~195,490 Stars. Validation sample would then contain:
* 176 EG sources
* 176 * 1130/376 = 530 YSOs
* 176 * 195 490/376 = 91 500 Stars (This is a bit over assuming of non star forming region, maybe do 50 000 instead.)
* Test on the whole sha-bang

Find best network this way
* Test Adam vs SGD?
* When on internet find another source?
* Maybe just train CLOUDS validate CORES?


Depending on output from this classifier, add it into the classifier, with flagging steps. If it's really good, automatically make this the most trusted classifier?

Make new flagging function


In regards to the very low JWST values, they are not off by a simple factor in all bands. Whatever is pushing them off is doing so as a function of wavelength.
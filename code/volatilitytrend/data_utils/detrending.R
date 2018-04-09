rm(list = ls())
library(genlasso)


cat('\014')
df=read.csv('/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/globe/1960-01-01_to_2010-12-31_data_avg_sub_north.csv',
            header = F)
x=data.matrix(df)
rm(df)

detrended=matrix(0,nrow=dim(x)[1],ncol=dim(x)[2])

for (i in 1:dim(x)[1]){
  print(paste0(Sys.time(),'...detrending location ',i,' of ',dim(x)[1]))
  y=x[i,]
  out=trendfilter(y,ord=1)
  cv=cv.trendfilter(out,k=3)
  detrended[i,]=y-out$beta[,cv$i.1se]
}

write.csv(detrended,'/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/globe/1960-01-01_to_2010-12-31_data_avg_sub_north_detrended.csv',row.names=FALSE)


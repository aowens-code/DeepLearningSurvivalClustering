#Insert path name to bottleneck for best run according to log-rank p-value
bottleneck <- read.csv("")
#Insert path name to clinical file
clinical <-read.csv("")
#Select time and event columns from clinical file
clinical <- clinical[c(2,3)]
newdf <- c()
for (i in 2:101){
  combo <- cbind(clinical, bottleneck[i])
  colnames(combo) <- c("Time","Event","Bottleneck")
  cox <- coxph(Surv(Time, Event) ~ Bottleneck, data = combo)
  x <- summary(cox)
  HR <-signif(x$coef[2], digits=3);
  HRLower <- signif(x$conf.int[,"lower .95"], 3)
  HRUpper <- signif(x$conf.int[,"upper .95"],3) 
  z = signif(x$coefficients[4], 3)
  zPValue <- signif(x$coefficients[5], 3)
  HRCI <- paste0(" (", HRLower, " - ", HRUpper, ")")
  res<-c(i-1, HR, HRCI, z, zPValue)
  names(res)<-c("Bottleneck Feature", "HR", "(95% CI)", "z", 
                "Pr(>|z|) ")
  newdf <- rbind(newdf, res)
}

#Insert path to write out file
write.csv(newdf,"", row.names=FALSE)

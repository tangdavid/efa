# read in pheno and covar
# write residuals to table with comment model in header?

get_resid <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  pheno <- read.table(args[1], header=T)

  covar <- read.table(args[2], header=T)
  joint <- merge(pheno, covar)
  joint[joint == "NA"] <- NA

  outfile <- args[3]
  
  # set batch to categorical if included
  if ("X22000.0.0" %in% colnames(joint)) {
    joint$X22000.0.0 <- as.factor(joint$X22000.0.0)
  }

  fit <- lm(joint[,3] ~ ., data=joint[,-c(1,2,3)], na.action=na.exclude)
  #fit <- lm(joint[,3] ~ joint[,4:ncol(joint)], na.action=na.exclude)
  e <- resid(fit)

  out <- cbind(joint[,1:2], e)
  out_trim <- na.omit(out)

  write.table(out_trim, outfile, row.names = F, sep = " ", quote=F)
}

get_resid()

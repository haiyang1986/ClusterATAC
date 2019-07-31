library(optparse)
library(survival)
#library(survminer)
#library(survcomp)
#library(KMsurv)
options(stringsAsFactors=FALSE)
option_list = list(
  make_option(c("-d", "--ATAC_data"), action="store", default='./TCGA_ATAC_peak_Log2Counts_dedup_sample.clusteratac',
              type='character', help="RNA_data"),
  make_option(c("-s", "--survival_file"), action="store", default='./clinical_PANCAN_patient_with_followup.tsv.clinical',
              type='character', help="clinical_file"),
  make_option(c("-f", "--final_save"), action="store", default='all.score',
              type='character', help="Path to genelist")
)
opt = parse_args(OptionParser(option_list=option_list))
atac_data <- read.table(opt$ATAC_data, check.names = FALSE, header = TRUE)
atac_sample <- read.table(opt$survival_file, check.names = FALSE, header = TRUE)
atac_sample$label <- atac_data$label + 1
sdf <- survdiff(Surv(atac_sample$days, atac_sample$status)~atac_data$label)
pvalue1<-paste("Log rank test p=",signif(1 - pchisq(sdf$chisq, length(sdf$n) - 1),2),"\n",sep="")
cat(pvalue1)

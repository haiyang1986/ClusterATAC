library(optparse)
library(survival)
#library(survminer)
#library(survcomp)
#library(KMsurv)
options(stringsAsFactors = FALSE)
option_list = list(
make_option(c("-d", "--ATAC_data"), action = "store", default = './TCGA_ATAC_peak_Log2Counts_dedup_sample.clusteratac',
type = 'character', help = "RNA_data"),
make_option(c("-s", "--survival_file"), action = "store", default = '/data/clinical_PANCAN_patient_with_followup.tsv.clinical',
type = 'character', help = "clinical_file"),
make_option(c("-m", "--method"), action = "store", default = 'Kmeans',
type = 'character', help = "Path to genelist"),
make_option(c("-o", "--output"), action = "store", default = './score.out',
type = 'character', help = "Path to genelist")
)
opt = parse_args(OptionParser(option_list = option_list))
atac_data <- read.table(opt$ATAC_data, check.names = FALSE, header = TRUE)
atac_sample <- read.table(opt$survival_file, check.names = FALSE, header = TRUE)
atac_sample$label <- atac_data$label + 1
sdf <- survdiff(Surv(atac_sample$days, atac_sample$status) ~ atac_data$label)
pvalue1 <- paste(opt$method, "    Log rank test p=", signif(1 - pchisq(sdf$chisq, length(sdf$n) - 1), 2), sep = "")
write(pvalue1,file=opt$output,append=TRUE)

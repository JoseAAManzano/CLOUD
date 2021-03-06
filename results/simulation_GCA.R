library(lmerTest)
library(ggplot2)
library(MuMIn)
library(sjPlot)
library(sjlabelled)
library(sjmisc)
library(emmeans)
library(Hmisc)

rm(list=ls())
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

simul <-
  read.table(
    'simulation_results.csv',
    sep=',',
    header=T,
    fileEncoding='utf-8',
    stringsAsFactors=TRUE
  )

simul$Version <- simul$Group

prod <- simul[simul$type == 'prod',]
reco <- simul[simul$type == 'reco',]

contrasts(prod$Version) <- contr.helmert(3)
contrasts(prod$label) <- c(-1, 1)

contrasts(reco$Version) <- contr.helmert(3)
contrasts(reco$label) <- c(-1, 1)

prod$Score <- prod$acc / 100
reco$Score <- reco$acc / 100

#### RECO

reco_simul <- aggregate(Score ~ run + Version + block + label, FUN=sum, data=reco)
t <- poly(unique(as.numeric(reco_simul$block)), 2)
reco_simul[,paste("ob", 1:2, sep='')] <- t[reco_simul$block, 1:2]
reco_simul$run <- as.factor(reco_simul$run)
reco_simul$anon <- as.factor(paste(reco_simul$Version, reco_simul$run, sep='_'))

m_simul_reco <- glmer(cbind(Score, 48-Score) ~ (ob1+ob2) * Version * label +
                        (ob1|anon:Version:label),
                      data=reco_simul,
                      control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=10000000)),
                      family='binomial')
m_simul_reco <- update(m_simul_reco, ~.-ob1:Version:label - ob2:Version:label)

summary(m_simul_reco)
r.squaredGLMM(m_simul_reco)
#confint(m_simul_reco, level=0.99)

lab <- c('Linear', 'Quadratic', 'Condition', 'ES-EN vs. ES-EU', 'MONO vs. BIL')
names(lab) <- c('ob1', 'ob2', 'label1', 'Version1', 'Version2')

rmterms <- c('ob1:Version1', 'ob1:Version2', 'ob2:Version1', 'ob2:Version2',
             'ob1:label1', 'ob2:label1', 'Version1:label1', 'Version2:label1')

plot_model(m_simul_reco, show.values=T, value.offset=.3, title="",
            rm.terms=rmterms, axis.labels=lab, transform=NULL, ci.lvl=0.99,
           dot.size=3, line.size=1.5, value.size=5, colors='bw') +
  font_size(labels.x=15, labels.y=15, axis_title.x=15) + ylim(c(-2.5, 12))

comp2 <- data.frame(reco_simul, Pred = fitted(m_simul_reco))

ggplot(comp2, aes(block, Score/48, color=Version, group=Version)) +
  facet_wrap(~ label) + 
  stat_summary(fun.data=mean_cl_normal, geom='pointrange',
               fun.args=list(conf.int=0.99)) +
  stat_summary(aes(y=Pred, group=Version),
               fun=mean, geom='line') +
  ylab('Accuracy \u00B1 99% CI') +
  xlab('Block')+
  ylim(0, 1)+
  theme_bw() + theme(text = element_text(size = 20))

#### PROD

prod_simul <- aggregate(Score ~ run + Version + block + label, FUN=mean, data=prod)
t <- poly(unique(as.numeric(prod_simul$block)), 2)
prod_simul[,paste("ob", 1:2, sep='')] <- t[prod_simul$block, 1:2]
prod_simul$anon <- as.factor(paste(prod_simul$Version, prod_simul$run, sep='_'))

m_simul_prod <- lmer(Score ~ (ob1 + ob2) * Version * label +
             (1|anon:Version:label),
           data=prod_simul,
           control=lmerControl(optimizer = "bobyqa", optCtrl=list(maxfun=1000000)),
           REML=F)
m_simul_prod <- update(m_simul_prod, ~.-ob1:Version:label - ob2:Version:label)

options(scipen=99)

summary(m_simul_prod, ddf='Satterthwaite')
r.squaredGLMM(m_simul_prod)
#confint(m_simul_prod, level=0.99)

plot_model(m_simul_prod, show.values=T, value.offset=.3, title="",
           axis.labels = lab, rm.terms=rmterms, ci.lvl=0.99,
           dot.size=3, line.size=1.5, value.size=5, colors='bw') +
  font_size(labels.x=15, labels.y=15, axis_title.x=15) + ylim(c(-0.12, 0.35))

comp2 <- data.frame(prod_simul, Pred = fitted(m_simul_prod))

ggplot(comp2, aes(block, Score, color=Version, group=Version)) +
  facet_wrap(~ label) + 
  stat_summary(fun.data=mean_cl_normal, geom='pointrange',
               fun.args=list(conf.int=0.99)) +
  stat_summary(aes(y=Pred, group=Version),
               fun.data=mean_cl_normal, geom='ribbon',
               fun.args=list(conf.int=0.99),
               alpha=0.2) +
  ylab('Accuracy \u00B1 99% CI') +
  xlab('Block') +
  ylim(0, 1)+
  theme_bw() + theme(text = element_text(size = 20))+
  theme(plot.title = element_text(hjust = 0.5))
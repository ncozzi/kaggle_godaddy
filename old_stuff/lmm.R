# install.packages("nlme")
# install.packages('reshape2')
# install.packages('psych')
# install.packages('rgdal')
# install.packages('spdep')
# install.packages('tmap')
# install.packages('ggmap')
# install.packages('raster')


setwd('C:/Users/ncozz/Documents/Python Scripts/kaggle_godaddy/kaggle_godaddy')
library(dplyr) # for random sample by group
library(reshape2)
library(tidyr)
library("lubridate")
library(GGally)
library(psych) # for pairs.panel
library(car)
library(panelr)
library(plm)
library(lme4)
library(lmerTest)
library('MuMIn')
library(data.table)

# spatial regression
library('sf')
library('spdep')
library(ggplot2)
library(usmap)
library(nlme)
library(varycoef)
library(parallel)
# library(clusterApply)

#
require(foreign)
require(MASS)



# bayesian
library("bayesplot")
library("ggplot2")
library("rstanarm")
library(bsreg)
library(stochvol)
library(adespatial)
# library(bmstdr)



# library(splines)
# library(glmmLasso)
# library(plot3D)
# library(lmerTest)


# https://stats.oarc.ucla.edu/r/faq/how-do-i-model-a-spatially-autocorrelated-outcome/

# http://www.geo.hunter.cuny.edu/~ssun/R-Spatial/spregression.html

# https://gdsl-ul.github.io/san/spatio-temporal-analysis.html
# https://gdsl-ul.github.io/san/spatio-temporal-analysis.html
# https://gdsl-ul.github.io/san/spatio-temporal-analysis.html
# https://gdsl-ul.github.io/san/spatio-temporal-analysis.html
# https://gdsl-ul.github.io/san/spatio-temporal-analysis.html
# https://gdsl-ul.github.io/san/spatio-temporal-analysis.html
# https://gdsl-ul.github.io/san/spatio-temporal-analysis.html
# https://gdsl-ul.github.io/san/spatio-temporal-analysis.html

# setwd('setwd(dirname(rstudioapi::getActiveDocumentContext()$path))')

train <- read.csv('data/train.csv')
test <- read.csv('data/test.csv')
census_starter <- read.csv('data/census_starter.csv')


################
# cleaning data

#train$cfips <- factor(train$cfips)
train$first_day_of_month <- as.Date(train$first_day_of_month, format = "%Y-%m-%d")
train$log.mbd <- log(train$microbusiness_density+0.1)
#test$cfips <- factor(test$cfips)
test$first_day_of_month <- as.Date(test$first_day_of_month, format = "%Y-%m-%d")


# census data: averages
census_df <- melt(census_starter, id.vars = c("cfips"))
census_df$variable <- as.character(census_df$variable)   
census_df$year <- as.integer(substr(census_df$variable, nchar(census_df$variable)-3, nchar(census_df$variable)))
census_df$variable <- substring(census_df$variable, 1, nchar(census_df$variable)-5)
census_df <- aggregate(census_df$value, list(census_df$cfips, census_df$variable), FUN=mean, na.rm=TRUE) 
colnames(census_df) <- c('cfips', 'variable', 'value')
census_df <- spread(census_df, variable, value)


# lag variables
train <- train %>%
  group_by(cfips) %>%
  mutate(mbd_lag1 = dplyr::lag(microbusiness_density, n = 1, default = NA))

train <- train %>%
  group_by(cfips) %>%
  mutate(diff_mbd = microbusiness_density - mbd_lag1)



train <- train %>%
  group_by(cfips) %>%
  mutate(log.mbd_lag = dplyr::lag(log.mbd, n = 1, default = NA))

train <- train %>%
  group_by(cfips) %>%
  mutate(diff_logmbd = log.mbd - logmbd_lag)



train <- train %>%
  group_by(cfips) %>%
  mutate(lag.year.logmbd = dplyr::lag(log.mbd, n = 12, default = NA))

train <- train %>%
  group_by(cfips) %>%
  mutate(lag.m.gwt.logmbd = dplyr::lag(log.mbd, n = 13, default = NA)-lag.year.logmbd)

train <- train %>%
  group_by(cfips) %>%
  mutate(lag.q.gwt.logmbd = dplyr::lag(log.mbd, n = 15, default = NA)-lag.year.logmbd)


train <- as.data.frame(train)
train %>%
  summarise_all(funs(sum(is.na(.))))

train <- na.omit(train)

# unit root tests

purtest(microbusiness_density~1, data=train, index = c("cfips"), pmax = 5, test = "madwu")
purtest(diff_mbd~1, data=train, index = "cfips", pmax = 2)
purtest(log.mbd~1, data=train, index = "cfips", pmax = 2)
purtest(diff_logmbd~1, data=train, index = "cfips", pmax = 1, test='madwu')


# adding lagged variables to test dataset
head(test)
dates.lag <- train$first_day_of_month[test$first_day_of_month %m+% years(-1)]
colnames(train)
test$state <- train[dates.lag,'state']
test$lag.year.logmbd <- train[dates.lag,'log.mbd']
test$lag.m.gwt.logmbd <- train[dates.lag,'log.mbd'] - 
  train[train$first_day_of_month[test$first_day_of_month %m+% years(-1) %m+% months(-1)], 'log.mbd']
test$lag.q.gwt.logmbd <- train[dates.lag,'log.mbd'] - 
  train[train$first_day_of_month[test$first_day_of_month %m+% years(-1) %m+% months(-3)], 'log.mbd']


#####
# merging with census data

train <- train %>%
  merge(y=census_df, by='cfips', all.x=TRUE)
test <- test %>%
  merge(y=census_df, by='cfips', all.x=TRUE)





# writing files
write.csv(train, "data/train_r.csv", row.names=FALSE)
write.csv(test, "data/test_r.csv", row.names=FALSE)










##################################

train <- read.csv('data/train_r.csv')
test <- read.csv('data/test_r.csv')

# spatial data
us_shape <- read_sf(dsn = "data/us-county-boundaries.shp")

us_shape$fips <- paste(us_shape$statefp, us_shape$countyfp, sep='')
us_shape$fips <- as.integer(us_shape$fips)
us_shape$intptlat <- as.numeric(us_shape$intptlat)
us_shape$intptlon <- as.numeric(us_shape$intptlon)
us_shape$mbd <- train$microbusiness_density[match(us_shape$fips, train$cfips)]

# merging with spatial data
train <- train %>%
  merge(y=us_shape[,c('countyfp','fips','intptlat','intptlon','geometry')], by.x='cfips', by.y='fips', all.x=TRUE)
test <- test %>%
  merge(y=us_shape[,c('countyfp','fips','intptlat','intptlon','geometry')], by.x='cfips', by.y='fips', all.x=TRUE)


colnames(train)[grep("intptlon", colnames(train))] <- 'lon'
colnames(train)[grep("intptlat", colnames(train))] <- 'lat'
colnames(test)[grep("intptlon", colnames(test))] <- 'lon'
colnames(test)[grep("intptlat", colnames(test))] <- 'lat'

train_df <- as.data.frame(train)
test_df <- as.data.frame(test)

# plot spatial data
train_plot <- aggregate(train$microbusiness_density,
                        list(train$cfips, train$intptlon, train$intptlat, train$state), FUN=mean, na.rm=TRUE)
colnames(train_plot) <- c('fips','lon','lat','state','mbd_avg')
train_plot$fips <- as.character(train_plot$fips)
train_plot$log_mbd_avg <- log(train_plot$mbd_avg)

plot_usmap(
  regions='counties',
  data = train_plot, values = "log_mbd_avg",
  # include = c("CA","NV","AZ","UT"), color='black') +
  # include = c("NC","SC","GA"), color='black') +
  include = c("TX"), color='black') +
  scale_fill_continuous(low = "lightyellow", high = "darkblue",
                        name = "log-average MB density, 2019-2021", label = scales::comma) + 
  labs(title = "Texas") +
  theme(panel.background = element_rect(color = "darkblue"))

train_sf <- st_as_sf(train[train$state=='Texas',], sf_column_name='geometry')

# rook neighbors
# poly2nb(train_sf, queen = FALSE)

# Moran's I test
# https://rspatial.org/analysis/3-spauto.html

train_sf <- st_as_sf(train[train$state=='Missouri',], sf_column_name='geometry')
wm <- terra::adjacent(train_sf, pairs=FALSE)
train_moran <- train[train$state=='Missouri',]
n <- dim(train_moran)[1]

y <- train_moran$log.mbd
ybar <- mean(y)

dy <- y - ybar
g <- expand.grid(dy, dy)
yiyj <- g[,1] * g[,2]

pm <- matrix(yiyj, ncol=n)
pmw <- pm * wm
pmw



# train_sf %>% spdep::poly2nb(c('geometry')) %>%
#   spdep::nb2listw(zero.policy = TRUE) -> train_sf_list
# 
# nycNbList %>%
#   spdep::moran.test(nycDat$UNEMP_RATE, ., zero.policy = TRUE)






# plot panel data

colsplot = c('cfips' ,'first_day_of_month', 'median_hh_inc', 'pct_bb', 'pct_college',
             'pct_foreign_born', 'pct_it_workers','log.mbd' )
df_plot <- train[,colsplot] %>% filter(cfips %in% sample(levels(cfips),50))
# ggpairs(df_plot, mapping = aes(color = cfips), cardinality_threshold=4000,
#         lower=list(continuous="smooth"),
#         diag=list(continuous="bar"),
#         upper = list(continuous = wrap("cor", size = 2.5, binwidth=0.5)),
#         axisLabels='show')

# plots of panel data
train_panel <- panel_data(df_plot, id=cfips, wave=first_day_of_month)
train_pdf = pdata.frame(df_plot, index = c("cfips", "first_day_of_month"), drop.index=TRUE)

train_panel %>% 
  line_plot(log.mbd,
            add.mean = TRUE,
            alpha = 0.2)

train_panel %>% 
  line_plot(log.mbd,
            overlay=FALSE,
            add.mean = TRUE,
            #mean.function = "loess",
            alpha = 0.2)

pairs.panels(train_panel)









#######################
# models
train$time <- 12*year(train$first_day_of_month) + month(train$first_day_of_month) - 
  12*year(min(train$first_day_of_month)) - month(min(train$first_day_of_month))
train$median_hh_inc <- scale(train$median_hh_inc)
train$pct_bb <- scale(train$pct_bb)
train$pct_college <- scale(train$pct_college)
train$pct_foreign_born <- scale(train$pct_foreign_born)
train$pct_it_workers <- scale(train$pct_it_workers)
test$time <- 12*year(test$first_day_of_month) + month(test$first_day_of_month) - 
  12*year(min(test$first_day_of_month)) - month(min(test$first_day_of_month))
test$median_hh_inc <- scale(test$median_hh_inc)
test$pct_bb <- scale(test$pct_bb)
test$pct_college <- scale(test$pct_college)
test$pct_foreign_born <- scale(test$pct_foreign_born)
test$pct_it_workers <- scale(test$pct_it_workers)





lmefit <- lmer(log.mbd ~ first_day_of_month + (1|cfips) + #(first_day_of_month||cfips)+
                 lag.year.logmbd +lag.m.gwt.logmbd + lag.q.gwt.logmbd +
                 median_hh_inc+#(median_hh_inc||county)+
                 pct_bb+#(pct_bb||county)+
                 pct_college+#(pct_college||county)+
                 pct_foreign_born+#(pct_foreign_born||county)+
                 pct_it_workers,#+(pct_it_workers||county),
               data = train, control = lmerControl(calc.derivs = FALSE))



summary(lmefit)
# confint(lmefit)
vif(lmefit)

# GOF
r.squaredGLMM(lmefit)
tab_model(lmefit)
AIC(lmefit)

# residuals
# plm <- plm(log.mbd ~ first_day_of_month + (1|cfips) + #(first_day_of_month||cfips)+
#              lag.year.logmbd +lag.m.gwt.logmbd + lag.q.gwt.logmbd +
#              median_hh_inc+#(median_hh_inc||county)+
#              pct_bb+#(pct_bb||county)+
#              pct_college+#(pct_college||county)+
#              pct_foreign_born+#(pct_foreign_born||county)+
#              pct_it_workers,#+(pct_it_workers||county),
#            data = train, model = "within", effect = "individual")
# summary(plm)
# purtest(residuals(plm), index='ISO', pmax = 3, exo = "intercept", test = "levinlin")

res_fit <- residuals(lmefit)
# pwartest(plm)
qqnorm(res_fit)
shapiro.test(res_fit[1:5000])








##########
# geospatial correlation


train$sqrt_median_hh_inc <- scale(sqrt((train$median_hh_inc - attributes(train$median_hh_in)$`scaled:scale`) +
  attributes(train$median_hh_in)$`scaled:center`))


lme <- nlme(log.mbd ~ time + 
                 lag.year.logmbd +lag.m.gwt.logmbd + lag.q.gwt.logmbd +
                 median_hh_inc+#(median_hh_inc||county)+
                 pct_bb+#(pct_bb||county)+
                 pct_college+#(pct_college||county)+
                 pct_foreign_born,#(pct_foreign_born||county)+
                 #pct_it_workers,#+(pct_it_workers||county),
           random= ~cfips,
           start=rep(0, length(unique(train$cfips))),
               data = train)


###########
# https://stackoverflow.com/questions/18857443/specifying-a-correlation-structure-for-a-linear-mixed-model-using-the-ramps-pack

train$log.mbd
train$state <- as.factor(train$state)
test$state <- as.factor(test$state)
lme <- rlm(log.mbd ~ time + state +
              lag.year.logmbd +lag.m.gwt.logmbd + lag.q.gwt.logmbd +
              median_hh_inc+#(median_hh_inc||county)+
              pct_bb+#(pct_bb||county)+
              pct_college+#(pct_college||county)+
              pct_foreign_born,#(pct_foreign_born||county)+
            #pct_it_workers,#+(pct_it_workers||county),
           #correlation = corGaus(form = ~ lon + lat, nugget=TRUE),
            data = train)#, psi = psi.bisquare, init = "lts")

cor <- corHaversine(1, form = ~ lon + lat , mimic="corSpher")
mbd.gau <- update(lme, correlation = cor, method = "M")
AIC(mbd.gau)
summary(mbd.gau)
# confint(lmefit)
vif(mbd.gau)

# GOF
r.squaredGLMM(mbd.gau)
tab_model(mbd.gau)
AIC(mbd.gau)


res_fit <- residuals(mbd.gau)
# pwartest(plm)
qqnorm(res_fit)
qqline(res_fit) 
shapiro.test(res_fit[1:5000])
train$residuals <- res_fit


wieghts_check <- train[train$state=='Oregon']
hweights <- data.frame(state=train$state, log.mbd = train$log.mbd, resid = res_fit, weight = mbd.gau$w)
hweights2 <- hweights[order(mbd.gau$w), ]
hweights2[hweights$state=='California',]

# plot(train[train$state=="Oregon",'lag.year.logmbd'], train[train$state=="Oregon",'residuals'])

predictions <- exp(predict(mbd.gau, test))-0.1
predictions[predictions<0]

df_kaggle <- data.frame(test$row_id, predictions)
colnames(df_kaggle) <- c('row_id','microbusiness_density')
head(df_kaggle)
write.csv(df_kaggle, 'attempt_lmm.csv', row.names = FALSE)















####################
# BAYESIAN APPROACH

options(mc.cores = parallel::detectCores())

# bsreg
o <- set_options(set_SLX(delta_scale = .1))
m <- get_bslx(y = y, X = X, options = o)
s <- sample(m, n_burn = 1000) # modification i

W <- listw.candidates(train[,c('intptlon','intptlat')])

m <- bslx(log.mbd ~ time + 
            lag.year.logmbd +lag.m.gwt.logmbd + lag.q.gwt.logmbd +
            median_hh_inc+pct_college+pct_foreign_born,
          W = c(intptlon, intptlat), data = train)
plot(m) # not converged?
m <- bm(m, n_save = 5000)






# bmrs

grid <- expand.grid(unique(train$intptlon), unique(train$intptlat))
K <- nrow(grid)

distance <- as.matrix(dist(grid))
W <- array(0, c(K, K))
W[distance == 1] <- 1   

pr1geo <- get_prior(log.mbd ~ time + (1|cfips) +
                      lag.year.logmbd +lag.m.gwt.logmbd + lag.q.gwt.logmbd +
                      median_hh_inc+pct_college+pct_foreign_born +
                      car(W),
                 data = train, family = student())
pr1geo$prior[1] <- "uniform(-1,1)"
pr1geo$prior[2:6] <- "normal(0,10)"


# fit a CAR model
fit_geo <- brm(log.mbd ~ time + (1|cfips) +
             lag.year.logmbd +lag.m.gwt.logmbd + lag.q.gwt.logmbd +
             median_hh_inc+pct_college+pct_foreign_born +
             car(W), 
           data = dat, data2 = list(W = W),
           family = student()) 
summary(fit_geo)


pr1 <- get_prior(Score_scaled ~ 0 + time + (1|ISO) + arma(p=1, q=0, cov=TRUE) ,
                 data = pfi_pdf, family = gaussian())
pr1$prior[1] <- "uniform(-1,1)"
pr1$prior[2:6] <- "normal(0,10)"


fit_bayes1 <- brm(Score_scaled ~ 0+time + (1 | ISO) + arma(p=1, q=0, cov=TRUE) ,
                  data = pfi_pdf, family = gaussian(), chains = 3,
                  iter = 10000, warmup = 5000, prior=pr1)



prior_summary(fit_bayes3)
summary(fit_bayes3)
plot(fit_bayes3)
pp_check(fit_bayes3, type='dens_overlay')
diagnostic_file(fit_bayes3)
pairs(fit_bayes3)
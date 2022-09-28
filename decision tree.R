library(rpart); library(rpart.plot)
library(forecast); library(caret); library(e1071)

spaceship <- read.csv("Desktop/kaggle/spaceship/train.csv", na.strings=c("","NA"))
spaceship_1 <- read.csv("Desktop/kaggle/spaceship/test.csv", na.strings=c("","NA"))

spaceship <- spaceship[ , -c(1, 4, 13)] 
spaceship_1 <- spaceship_1[ , -c(4, 13)] 

spaceship<-na.omit(spaceship)

spaceship$Age[is.na(spaceship$Age)] <- mean(spaceship$Age, na.rm = TRUE)
spaceship$VRDeck[is.na(spaceship$VRDeck)] <- mean(spaceship$VRDeck, na.rm = TRUE)
spaceship$Spa[is.na(spaceship$Spa)] <- mean(spaceship$Spa, na.rm = TRUE)
spaceship$ShoppingMall[is.na(spaceship$ShoppingMall)] <- mean(spaceship$ShoppingMall, na.rm = TRUE)
spaceship$FoodCourt[is.na(spaceship$FoodCourt)] <- mean(spaceship$FoodCourt, na.rm = TRUE)
spaceship$RoomService[is.na(spaceship$RoomService)] <- mean(spaceship$RoomService, na.rm = TRUE)

spaceship_1$Age[is.na(spaceship_1$Age)] <- mean(spaceship_1$Age, na.rm = TRUE)
spaceship_1$VRDeck[is.na(spaceship_1$VRDeck)] <- mean(spaceship_1$VRDeck, na.rm = TRUE)
spaceship_1$Spa[is.na(spaceship_1$Spa)] <- mean(spaceship_1$Spa, na.rm = TRUE)
spaceship_1$ShoppingMall[is.na(spaceship_1$ShoppingMall)] <- mean(spaceship_1$ShoppingMall, na.rm = TRUE)
spaceship_1$FoodCourt[is.na(spaceship_1$FoodCourt)] <- mean(spaceship_1$FoodCourt, na.rm = TRUE)
spaceship_1$RoomService[is.na(spaceship_1$RoomService)] <- mean(spaceship_1$RoomService, na.rm = TRUE)


set.seed(123)  
train.index <- sample(c(1:dim(spaceship)[1]), dim(spaceship)[1]*0.7)  
train.df <- spaceship[train.index, ]
valid.df <- spaceship[-train.index, ]

full.tree <- rpart(Transported ~ ., data = train.df, method = "class", xval = 10, cp = 0, minsplit = 1)

#Number of leaves for full tree
length(full.tree$frame$var[full.tree$frame$var == "<leaf>"])

#find best cp
printcp(full.tree)

# pruned.tree
pruned.tree <- prune(full.tree, cp = 8.4266e-04)
length(pruned.tree$frame$var[pruned.tree$frame$var == "<leaf>"])


#accuracy
pruned_train<-predict(pruned.tree,train.df,type='class')
confusionMatrix(pruned_train, as.factor(train.df$Transported)) 
pruned_valid<-predict(pruned.tree,valid.df,type='class')
confusionMatrix(pruned_valid, as.factor(valid.df$Transported)) 

#upload
upload <- predict(pruned.tree,spaceship_1)[1:4277]
df<- data.frame(spaceship_1$PassengerId,upload)
names(df)<-c("PassengerId","Transported")
write.csv(df,file="tree2.csv",row.names = F)

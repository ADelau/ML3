#Load data
userData <- read.table("data_user.csv", sep= ",", header= TRUE)

zip_codes <- userData[, colnames(userData) == "zip_code"]

library(zipcode)
data(zipcode)

longitude <- vector("numeric", length(zip_codes))
latitude <- vector("numeric", length(zip_codes))

for(i in 1:length(zip_codes)){
  #weird zip code
  if(is.na(as.numeric(as.character(zip_codes[i])))){
    longitude[i] <- NA
    latitude[i] <- NA
  }
  else{
    index <- which.min(abs(as.numeric(as.character(zipcode[, colnames(zipcode) == "zip"])) - as.numeric(as.character(zip_codes[i]))))
    longitude[i] <- zipcode[index, colnames(zipcode) == "longitude"]
    latitude[i] <- zipcode[index, colnames(zipcode) == "latitude"]
  }
}

userData <- cbind(userData, longitude, latitude)
colnames(userData) <- cbind(colnames(userData), "longitude", "latitude")

write.table(userData, "data_user_2.csv", sep = ",", row.names = FALSE)

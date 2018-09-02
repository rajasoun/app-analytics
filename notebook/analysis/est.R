# Clean Up Environment 
rm(list=ls())

pkgs <- c("RMariaDB", "DBI",  "tidyverse","tictoc","pacman")

# Install Packages
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# Load Packages
lapply(pkgs, library, character.only = TRUE)

pacman::p_load_gh("trinker/qdapTools")


con <- dbConnect(RMariaDB::MariaDB(), 
                 host = "10.126.100.201", 
                 username = "root",
                 password = "secret",
                 dbname = "est",
                 port = 3306)

tables <- dbListTables(con)
tables

learning_path <- dbReadTable(con, "ep_learning_path")
dbListFields(con, "ep_learning_path")

sql_lp_row_cnt_null_url <- "SELECT count(*) FROM ep_learning_path"
sql_lp_row_cnt <- "SELECT count(*) FROM ep_learning_path WHERE launchURL IS NOT NULL"
sql_lp_rows <- "SELECT title, launchURL FROM ep_learning_path WHERE launchURL IS NOT NULL"

df <- dbGetQuery(con, sql_lp_row_cnt_null_url)
print("Total Records ")
df
df <- dbGetQuery(con, sql_lp_row_cnt)
print("Total Records excluding NULL Launch URLs")
df
df <- dbGetQuery(con, sql_lp_rows)
print("LP Rows")
dim(df)
head(df)


# Disconnect from the database
dbDisconnect(con)


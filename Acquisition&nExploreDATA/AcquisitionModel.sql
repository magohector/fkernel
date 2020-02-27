--Distributed training model OF SENSOR MODIS

create table  b20 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 200000000) T 			       
GROUP BY latitude, longitude;

create table b21  as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 210000000) T 			       
GROUP BY latitude, longitude;

create table b22 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 220000000) T 			       
GROUP BY latitude, longitude;

create table b23 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 230000000) T 			       
GROUP BY latitude, longitude;

create table b24 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 240000000) T 			       
GROUP BY latitude, longitude;

create table b25 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 250000000) T 			       
GROUP BY latitude, longitude;

create table b26 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 260000000) T 			       
GROUP BY latitude, longitude;

create table b27 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 270000000) T 			       
GROUP BY latitude, longitude;

create table  b28 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM


(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 280000000) T 			       
GROUP BY latitude, longitude;

create table b29 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 290000000) T 			       
GROUP BY latitude, longitude;

create table b30 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 300000000) T 			       
GROUP BY latitude, longitude;

create table b31 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 310000000) T 			       
GROUP BY latitude, longitude;

create table b32 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 320000000) T 			       
GROUP BY latitude, longitude;


create table b33 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 330000000) T 			       
GROUP BY latitude, longitude;

create table b34 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 340000000) T 			       
GROUP BY latitude, longitude;

create table b35 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 10000000 OFFSET 350000000) T 			       
GROUP BY latitude, longitude;

create table b36 as
SELECT latitude, longitude, sum(band1) as band1, sum(band2) as band2,   sum(band3) as band3, 
 sum(band4) as band4, sum(band5) as band5, sum(band6) as band6, sum(band7) as band7, count(*) cuenta
FROM
(SELECT * FROM bands1_7 LIMIT 13670476 OFFSET 360000000) T 			       
GROUP BY latitude, longitude;




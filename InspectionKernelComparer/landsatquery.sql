SELECT * FROM(
SELECT
  latitude, longitude, avg(band1) as band1, avg(band2) as band2,
   avg(band3) as band3, avg(band4) as band4, avg(band5) as band5,
	 avg(band6) as band6, avg(band7) as band7
FROM 
	reflectance
GROUP BY
	latitude, longitude LIMIT 100) as foo
JOIN
	irradiance_grid_450 
USING(latitude, longitude);

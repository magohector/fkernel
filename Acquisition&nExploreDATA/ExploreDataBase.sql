
--Diccionario
SELECT table_name,column_name,data_type  FROM information_schema.columns 
WHERE table_name in (SELECT table_name FROM information_schema.tables
WHERE table_schema='public' AND table_type='BASE TABLE' 
AND table_name='general')
ORDER BY table_name, column_name

--Extracción llaves foráneas y primarias
SELECT
    tc.table_schema, 
    tc.constraint_name, 
    tc.table_name, 
	tc.constraint_type,
    kcu.column_name, 
    ccu.table_schema  foreign_table_schema,
    ccu.table_name  foreign_table_name,
    ccu.column_name  foreign_column_name
	
FROM 
    information_schema.table_constraints AS tc 
    ,information_schema.key_column_usage AS kcu      
    ,information_schema.constraint_column_usage AS ccu
      
WHERE 
	  tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
	  AND ccu.constraint_name = tc.constraint_name
      AND ccu.table_schema = tc.table_schema
	  

AND tc.constraint_type = 'FOREIGN KEY' OR tc.constraint_type='PRIMARY KEY';
--AND tc.table_name='general';



--Extracción de datos numéricos sin llaves
SELECT table_name,column_name,data_type FROM information_schema.columns
WHERE  table_name in (
SELECT table_name FROM information_schema.tables
WHERE table_schema='public' AND table_type='BASE TABLE') 
AND column_name not in(
SELECT
    kcu.column_name
	
FROM 
    information_schema.table_constraints AS tc 
    ,information_schema.key_column_usage AS kcu      
    ,information_schema.constraint_column_usage AS ccu
      
WHERE 
	  tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
	  AND ccu.constraint_name = tc.constraint_name
      AND ccu.table_schema = tc.table_schema
	  

AND tc.constraint_type IN ('PRIMARY KEY','FOREIGN KEY'))AND data_type='numeric' 

--Cursor para extracción de modas
create or replace function mimoda() returns void as
$BODY$
DECLARE
	
	sql_stmt3 VARCHAR(500);
	reg3 RECORD;
    cur_moda CURSOR FOR 
    SELECT table_name,column_name,data_type  FROM information_schema.columns 
    WHERE table_name in (SELECT table_name FROM information_schema.tables
    WHERE table_schema='public' AND table_type='BASE TABLE') 
	AND column_name not in(
SELECT
    kcu.column_name
	
FROM 
    information_schema.table_constraints AS tc 
    ,information_schema.key_column_usage AS kcu      
    ,information_schema.constraint_column_usage AS ccu
      
WHERE 
	  tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
	  AND ccu.constraint_name = tc.constraint_name
      AND ccu.table_schema = tc.table_schema
		  

AND tc.constraint_type IN ('PRIMARY KEY','FOREIGN KEY')
AND tc.table_name='general')
AND table_name like 'general' 
    ORDER BY table_name, column_name;    
BEGIN 
	DELETE FROM modas;
	OPEN cur_moda;
	FETCH cur_moda into reg3;
	    
    WHILE(FOUND) LOOP
    	sql_stmt3 := 'INSERT INTO modas (name,moda)
		SELECT '''||reg3.column_name||''', T FROM
  		(SELECT '||reg3.column_name||', count(*) as c
         FROM general GROUP BY '||reg3.column_name||' HAVING count(*)>10) as T
  		ORDER BY c desc LIMIT 1';
		 
		EXECUTE   sql_stmt3;
        RAISE NOTICE '%',reg3;
    	FETCH cur_moda into reg3;
    END LOOP;
    CLOSE cur_moda;
    RETURN;
END 
$BODY$
LANGUAGE 'plpgsql'

--Función para obtención de frecuencias
create or replace function mifrecuencias() returns void as
$BODY$
DECLARE
	
	sql_stmt VARCHAR(500);
	reg RECORD;
    cur_frecu CURSOR FOR 
    SELECT table_name,column_name,data_type  FROM information_schema.columns 
    WHERE table_name in (SELECT table_name FROM information_schema.tables
    WHERE table_schema='public' AND table_type='BASE TABLE') AND
    table_name like 'ventas'    
    ORDER BY table_name, column_name;    
BEGIN 	
	DELETE FROM frecuencias;
	OPEN cur_frecu;
FETCH cur_frecu into reg;  
    WHILE(FOUND) LOOP
    	sql_stmt := 'INSERT INTO frecuencias(nombre,columna,frecuencia)       
					 SELECT '''||reg.column_name||''','||reg.column_name||', count(*) FROM ventas GROUP BY '||reg.column_name || ' HAVING COUNT(*)>2';
   		EXECUTE   sql_stmt;
        RAISE NOTICE '%',sql_stmt;
    	FETCH cur_frecu into reg;
    END LOOP;
    CLOSE cur_frecu;
    RETURN;
END 
$BODY$
LANGUAGE 'plpgsql'


--Cursor para obtención de estadísticos de calidad
create or replace function miestadistica() returns void as
$BODY$
DECLARE
	
	sql_stmt2 VARCHAR(1000);
	reg2 RECORD;
    cur_esta CURSOR FOR 
    SELECT table_name,column_name,data_type  FROM information_schema.columns 
    WHERE table_name in (SELECT table_name FROM information_schema.tables
    WHERE table_schema='public' AND table_type='BASE TABLE') 
	AND column_name not in(
SELECT
    kcu.column_name
	
FROM 
    information_schema.table_constraints AS tc 
    ,information_schema.key_column_usage AS kcu      
    ,information_schema.constraint_column_usage AS ccu
      
WHERE 
	  tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
	  AND ccu.constraint_name = tc.constraint_name
      AND ccu.table_schema = tc.table_schema
		  

AND tc.constraint_type IN ('PRIMARY KEY','FOREIGN KEY')
AND tc.table_name='general') and data_type = 'double precision'
AND table_name like 'general' 
    ORDER BY table_name, column_name;    
BEGIN 
	DELETE FROM estadisticas;
	OPEN cur_esta;
	FETCH cur_esta into reg2;
	    
    WHILE(FOUND) LOOP
    	sql_stmt2 := 'INSERT INTO estadisticas (nombre,min,max,avg,stddev,ran_min,
						ran_max, minX, maxX,Q1, medianQ2, Q3, IQR, count)
		(SELECT	nombre, count,min,max,avg,stddev_samp,ran_min,ran_max,
		Q1-1.5*IQR as minX, Q3+1.5*IQR as maxX, Q1, medianQ2, Q3, IQR
    	FROM
		(SELECT 
        nombre, Q3-Q1 as IQR, Q1 as Q1,medianQ2 as medianQ2,
	 	Q3 as Q3, min,max,avg,stddev_samp,ran_min,ran_max,count
    	FROM
		(SELECT '''||reg2.column_name||''' as nombre,min('||reg2.column_name||'),
		max('||reg2.column_name||') , AVG('||reg2.column_name||'),
		stddev_samp('||reg2.column_name||'), AVG('||reg2.column_name||')+
		stddev_samp('||reg2.column_name||') as ran_min,
		AVG('||reg2.column_name||')-stddev_samp('||reg2.column_name||') as ran_max, 
		count('||reg2.column_name||') as count,
    	percentile_disc(0.25) within  group (order by '||reg2.column_name||') as Q1,
        percentile_disc(0.5) within  group (order by '||reg2.column_name||') as medianQ2,
        percentile_disc(0.75) within  group (order by '||reg2.column_name||') as Q3
         FROM general )T)T2)';

   		EXECUTE   sql_stmt2;
        RAISE NOTICE '%',reg2;
    	FETCH cur_esta into reg2;
    END LOOP;
    CLOSE cur_esta;
    RETURN;
END 
$BODY$
LANGUAGE 'plpgsql'





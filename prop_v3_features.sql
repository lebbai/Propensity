---------------
---- LL Cross-Sell Modeling
---- 10.44.94.64
----------------

/*
All LL orders after Jan1st 2018
*/
drop table ll1;

CREATE MULTISET VOLATILE TABLE ll1
AS (
 SELECT
			NORTON_ACCOUNT_GUID,PROGRAM_ID, ORDER_DATE,SKU,PRODUCT_FAMILY,INCOMING_PSN,NET_SALES
		FROM
			CDCBVDB.BV_ESTORE_ORD ll
		LEFT JOIN
			CDCBVDB.BV_ITEM_EXT
				ON BV_ITEM_EXT.SKU_NUM = ll.SKU
		WHERE
			PROD_CNSMR like ('%LL%')
		AND
			ORDER_DATE >= '2018-01-01'
		AND
		   ORDER_STATUS = 'CLOSED'
	---	AND
	---		NET_SALES > 0
		  
QUALIFY ROW_NUMBER () OVER (PARTITION BY NORTON_ACCOUNT_GUID ORDER BY ORDER_DATE) = 1  --- first order date		   
		   ) WITH DATA PRIMARY INDEX (NORTON_ACCOUNT_GUID) ON COMMIT PRESERVE ROWS
    ; 
 
    
 select count(*) from ll1; ---61338
 select count(distinct NORTON_ACCOUNT_GUID) from ll1 where NET_SALES>0;---60971
 select top 10 * from ll1;
 select count(distinct NORTON_ACCOUNT_GUID) from ll1 where ORDER_DATE is null; ----0
 select min(ORDER_DATE) from ll1; ---2018-01-01
 
 
------------------------
---- find the active product for these GUIDS
---- psn| prod fmly | 
-------
 
drop table ll_prod;
create multiset volatile table ll_prod
as
(
select a.*,
	  PROD_FMLY_GRP
	  ,PSN
	  ,EXPIRATION_DATE
	  ,CURRENT_AR_FLAG
      ,PURCHASE_AR_FLAG
      ,BV_MT_REG_PSN_DTLS.ACQUISITION_CHANNEL as PSN_CHANNEL
      ,COALESCE (BV_MT_REG_PSN_DTLS.ORDER_DATE, BV_MT_REG_PSN_DTLS.START_DATE) PSN_ORDER_DATE
from ll1 a
     LEFT join CDCBVDB.BV_MT_REG_PSN_DTLS on a.NORTON_ACCOUNT_GUID = BV_MT_REG_PSN_DTLS.ACCOUNT_GUID
	   LEFT JOIN CDCBVDB.BV_ITEM_EXT ON BV_ITEM_EXT.SKU_NUM = BV_MT_REG_PSN_DTLS.SKUP
where BV_MT_REG_PSN_DTLS.ACQUISITION_CHANNEL <> 'CSP'
AND PROD_FMLY_GRP <> 'IDP_F'
AND IS_CORE = 'Y'
AND
PAID_STATUS = 'PAID'
AND
 IS_TEST_FL = 'N'
AND PSN_ORDER_DATE < a.ORDER_DATE 
QUALIFY ROW_NUMBER () OVER (PARTITION BY a.NORTON_ACCOUNT_GUID ORDER BY PSN_ORDER_DATE desc) = 1 --- recentmost order date
) with data on commit preserve rows;	---59312, missing 1659 GUID's 

select count(*) from ll_prod;---54089
---select top 10 * from ll_prod;
--- select 60971 - 59312;
-----------
--- 1659 GUIDS with first order as CORE
---
----------
drop table ll_prod1;
CREATE MULTISET VOLATILE TABLE ll_prod1
AS (
select a.*,
	  BV_ITEM_EXT.PROD_FMLY_GRP
	  ,BV_MT_REG_PSN_DTLS.PSN
	  ,BV_MT_REG_PSN_DTLS.EXPIRATION_DATE
	  ,BV_MT_REG_PSN_DTLS.CURRENT_AR_FLAG
      ,BV_MT_REG_PSN_DTLS.PURCHASE_AR_FLAG
      ,BV_MT_REG_PSN_DTLS.ACQUISITION_CHANNEL as PSN_CHANNEL,
      ----BV_MT_REG_PSN_DTLS.SKUP,
      COALESCE (BV_MT_REG_PSN_DTLS.ORDER_DATE, BV_MT_REG_PSN_DTLS.START_DATE) PSN_ORDER_DATE
from ll1 a
 LEFT join CDCBVDB.BV_MT_REG_PSN_DTLS on a.NORTON_ACCOUNT_GUID = BV_MT_REG_PSN_DTLS.ACCOUNT_GUID
	   LEFT JOIN CDCBVDB.BV_ITEM_EXT ON BV_ITEM_EXT.SKU_NUM = BV_MT_REG_PSN_DTLS.SKUP
	   	LEFT JOIN ll_prod ON a.NORTON_ACCOUNT_GUID = ll_prod.NORTON_ACCOUNT_GUID ---- take out guids from core_psn
WHERE ll_prod.NORTON_ACCOUNT_GUID IS NULL 
AND BV_MT_REG_PSN_DTLS.ACQUISITION_CHANNEL <> 'CSP'
AND BV_ITEM_EXT.PROD_FMLY_GRP <> 'IDP_F'
AND IS_CORE = 'Y'
AND
PAID_STATUS = 'PAID'
AND
 IS_TEST_FL = 'N' 
AND 
 a.NORTON_ACCOUNT_GUID not in (select distinct NORTON_ACCOUNT_GUID from ll_prod)
 
QUALIFY ROW_NUMBER () OVER (PARTITION BY a.NORTON_ACCOUNT_GUID ORDER BY BV_MT_REG_PSN_DTLS.EXPIRATION_DATE) = 1	) WITH DATA ON COMMIT PRESERVE ROWS; 

select count(*) from ll_prod1; ---1933

------------
-- 
/*
 NORTON_ACCOUNT_GUID,
 PROGRAM_ID, 
 ORDER_DATE,
 SKU,
 PRODUCT_FAMILY,INCOMING_PSN,NET_SALES, 
 PROD_FMLY_GRP,
 PSN,
 EXPIRATION_DATE,
 CURRENT_AR_FLAG,
 PURCHASE_AR_FLAG,
 PSN_CHANNEL,
 SKUP,
 PSN_ORDER_DATE,
 0 as FIRST_PURCHASE_FL
 */
------------
-------------
-- UNION THE ABOVE TWO TABLES
-------------
------------

drop table ll2;
create multiset volatile table ll2
as
(
select * from
(select
	NORTON_ACCOUNT_GUID
	,PROGRAM_ID 
	,ORDER_DATE
	,INCOMING_PSN
	,NET_SALES
	,PROD_FMLY_GRP
 	,PSN
 	,EXPIRATION_DATE
 	,CURRENT_AR_FLAG
 	,PURCHASE_AR_FLAG
 	,PSN_CHANNEL
	,SKU
 	,PSN_ORDER_DATE
 	,0 as FIRST_PURCHASE_FL
from ll_prod
where NORTON_ACCOUNT_GUID is not null
and PSN is not null) t1
UNION
(select
NORTON_ACCOUNT_GUID,
PROGRAM_ID, 
ORDER_DATE,
INCOMING_PSN,
NET_SALES,
PROD_FMLY_GRP,
 PSN,
 EXPIRATION_DATE,
 CURRENT_AR_FLAG,
 PURCHASE_AR_FLAG,
 PSN_CHANNEL,
 SKU,
 PSN_ORDER_DATE,
 1 as FIRST_PURCHASE_FL
from ll_prod1
where NORTON_ACCOUNT_GUID is not null
and PSN is not null)
)with data on commit preserve rows;

select count(*) from ll2; --- 61245

select * from ll2; --- 28631


/*
select count(*) from dl_crm.jgray_customer_funnel_passive;
select count(distinct ACCOUNT_GUID) from dl_crm.jgray_customer_funnel_passive;
*/
-------------------------------
/*
 * Add LL_FL as 1
 */
-------------------------------

drop table temp_ll;
create multiset volatile table temp_ll
as
(
select a.*,
       1 as LL_FL
from ll2 a)  with data on commit preserve rows;  


----------------------------------------------------------------------
----------------------------------------------------------------------
--- Get All the norton US, Active, Paid users who have not bought LL
---      ACQUISITION_CHANNEL,
---      CURRENT_AR_FLAG,
---      PURCHASE_AR_FLAG
----------------------------------------------------------------------
----------------------------------------------------------------------
/*
 NORTON_ACCOUNT_GUID,
PROGRAM_ID, 
ORDER_DATE,
INCOMING_PSN,(dont need)
NET_SALES, (dont need)
PROD_FMLY_GRP,
 PSN,
 EXPIRATION_DATE,
 CURRENT_AR_FLAG,
 PURCHASE_AR_FLAG,
 PSN_CHANNEL,
 SKU,
 PSN_ORDER_DATE,
 */

drop table temp1;
create multiset volatile table temp1
as
(
SELECT a1.*,
       0 as LL_FL,
       -1 as FIRST_PURCHASE_FLAG
FROM
(SELECT
	BV_MT_REG_PSN_DTLS.PSN,
	BV_MT_REG_PSN_DTLS.SKUP,
    --- REGS_USER_EML_ADDR as EMAIL,
	BV_MT_REG_PSN_DTLS.ACCOUNT_GUID,
	BV_MT_REG_PSN_DTLS.EXPIRATION_DATE,
	BV_MT_REG_PSN_DTLS.ACQUISITION_CHANNEL as PSN_CHANNEL,
	CURRENT_AR_FLAG,
	PURCHASE_AR_FLAG,
	'2020-01-01' as ORDER_DATE,
	'NON_LL_PROG' as PROGRAM_ID,
	COALESCE (BV_MT_REG_PSN_DTLS.ORDER_DATE, BV_MT_REG_PSN_DTLS.START_DATE) PSN_ORDER_DATE,
    BV_ITEM_EXT.PROD_FMLY_GRP

FROM
	CDCBVDB.BV_MT_REG_PSN_DTLS 
	
LEFT JOIN
	CDCBVDB.BV_ITEM_EXT 
		ON BV_ITEM_EXT.SKU_NUM = BV_MT_REG_PSN_DTLS.SKUP

INNER JOIN
	(
	SELECT
		REGS_ACCT_GUID,
		REGS_CTRY_ISO2_CD,
		REGS_USER_EML_ADDR
	FROM
		CDCBVDB.BV_REGS_USER
	WHERE
		cast(ETL_END_DTTM as date) = '9999-12-31'
	)BV_REGS_USER 
		ON BV_REGS_USER.REGS_ACCT_GUID = BV_MT_REG_PSN_DTLS.ACCOUNT_GUID
		
WHERE
BV_MT_REG_PSN_DTLS.EXPIRATION_DATE > '2018-01-01' --- active as of Jan 2018
	
AND
	BV_MT_REG_PSN_DTLS.COUNTRY_CODE = 'US'	
AND
	REGS_CTRY_ISO2_CD = 'US'
AND
	PAID_STATUS = 'PAID'
AND
	ACQUISITION_CHANNEL IN ('ONLINE','RETAIL')
AND
	ACTIVATION_DATE is not null
AND
	BV_MT_REG_PSN_DTLS.ACCOUNT_GUID is not null
AND
	BV_ITEM_EXT.PROD_FMLY_GRP IN ('NIS','N360','NS','NS-BU','NAV','N360-MD','NS-S','NS-SP','N360-PE','NAVB')
AND
	CURRENT_FLAG = 'Y'
--AND
	--REGS_USER_EML_ADDR not like ('%symantec.com')
AND BV_MT_REG_PSN_DTLS.ACQUISITION_CHANNEL <> 'CSP'
AND BV_ITEM_EXT.PROD_FMLY_GRP <> 'IDP_F'
AND IS_CORE = 'Y'	
AND
 IS_TEST_FL = 'N' 
---AND
---	ACCOUNT_GUID not in
---		(select distinct NORTON_ACCOUNT_GUID  from core_psn)

	AND ---- ACER exclusion
	NOT EXISTS 
		(
		SELECT DISTINCT
			BV_MT_REG_PSN_DTLS.ACCOUNT_GUID
		FROM
			CDCBVDB.BV_MT_REG_PSN_DTLS ACER
		LEFT JOIN
			CDCBVDB.BV_ITEM_EXT	
				ON BV_ITEM_EXT.SKU_NUM = ACER.SKUP
		WHERE
			PROD_FMLY_GRP LIKE ('NS%')
		AND
			PTNR_GRP like ('%ACER%')
		AND
			BV_MT_REG_PSN_DTLS.EXPIRATION_DATE > '2018-01-01'
		AND 
			ACER.ACCOUNT_GUID = BV_MT_REG_PSN_DTLS.ACCOUNT_GUID
		) 
	QUALIFY ROW_NUMBER () OVER (PARTITION BY ACCOUNT_GUID ORDER BY EXPIRATION_DATE desc) = 1 )a1
---- lifelock exclusion	
LEFT JOIN temp_ll ON a1.ACCOUNT_GUID = temp_ll.NORTON_ACCOUNT_GUID
WHERE temp_ll.NORTON_ACCOUNT_GUID IS NULL
		) with data primary index (PSN,ACCOUNT_GUID) on commit preserve rows; ---8276845
		
select count(distinct ACCOUNT_GUID) from temp1;  ---8425951

select count(*) from temp1; ---8425951


----------------------------------------------------------------------
----------------------------------------------------------------------
/*
 * Scrub the list further using LL_Match
 * 
 */
----------------------------------------------------------------------
----------------------------------------------------------------------
drop table temp2;

create multiset volatile table temp2
as(
select a.*
from temp1 a
left join 
(SELECT
		ACCOUNT_GUID,
		MAX(LL_ACTIVE_FLG) LL_ACTIVE_FLG
	FROM
		DL_CRM.LL_MATCH2
	WHERE
		LL_ACTIVE_FLG = '1'
	GROUP BY
		ACCOUNT_GUID
	) LL_MATCH2
ON cast(LL_MATCH2.ACCOUNT_GUID as float) = a.ACCOUNT_GUID
where LL_MATCH2.ACCOUNT_GUID is null) with data primary index (PSN,ACCOUNT_GUID) on commit preserve rows;

select count(distinct ACCOUNT_GUID) from temp2; ---7847873






----------------------------------------------------------------------
----------------------------------------------------------------------
--- Union Core and Non-Core Users to create base table
----------------------------------------------------------------------
/*
---
ACCOUNT_GUID,
PSN,
SKUP,
TO_DATE(CORE_ORDER_DATE1) CORE_ORDER_DATE1,
ACQUISITION_CHANNEL,
PROD_FMLY_GRP,
CURRENT_AR_FLAG,
PURCHASE_AR_FLAG,
PSN_CHANNEL,
EXPIRATION_DATE,
PSN_ORDER_DATE,
FIRST_PURCHASE_FLAG,
CORE_FL
		 */
select top 10 * from temp_ll;		
----------------------------------------------------------------------
----------------------------------------------------------------------
----------------------------------------------------------------------

drop table base1;
CREATE MULTISET VOLATILE TABLE base1
as (
select  * from
(
select  ACCOUNT_GUID,
		PSN,
		SKUP ,
		TO_DATE(ORDER_DATE) ORDER_DATE,
		PROD_FMLY_GRP,
		CURRENT_AR_FLAG,
		PURCHASE_AR_FLAG,
		PSN_CHANNEL,
		EXPIRATION_DATE,
		PSN_ORDER_DATE,
		FIRST_PURCHASE_FLAG,
		PROGRAM_ID,
		LL_FL

from temp2
) t1
UNION
(
select  NORTON_ACCOUNT_GUID as ACCOUNT_GUID,
		PSN,
		SKU as SKUP,
		ORDER_DATE,
		PROD_FMLY_GRP,
		CURRENT_AR_FLAG,
		PURCHASE_AR_FLAG,
		PSN_CHANNEL,
		EXPIRATION_DATE,
		PSN_ORDER_DATE,
		FIRST_PURCHASE_FL as FIRST_PURCHASE_FLAG,
		PROGRAM_ID,
		LL_FL
from temp_ll
)
) with data on commit preserve rows ;

select count(*) from base1; ---7768015

select count(distinct(ACCOUNT_GUID)) from base1; --- 7768015

select count(*),LL_FL from base1 group by LL_FL; --- {1:28631, 0:7739384}

select avg(LL_FL) from base1; ---0.0036857549837378016


-------------------
/*
 * Modify Current AR Flag
 * 
 */
-------------------
-----------------

drop table base;
create multiset volatile table base
as
(
SELECT
	 a.*
	--,EBE_AR_HIST_SER_NUM AS PSN
	,b.EBE_AR_HIST_CR_DTTM AS OPT_OUT_DATE_AR_HIST
	,b.EBE_AR_HIST_OPT_IN_VAL AS OPT_OUT_VALUE_AR_HIST
FROM base1 a
left join CDCBVDB.BV_EBE_AR_HIST b on a.PSN = b.EBE_AR_HIST_SER_NUM and b.EBE_AR_HIST_CR_DTTM < a.ORDER_DATE
QUALIFY ROW_NUMBER () OVER (PARTITION BY PSN ORDER BY OPT_OUT_DATE_AR_HIST desc) = 1
)  with data on commit preserve rows;  


--- save this to a permanent table
/*
drop table dl_crm.mgua_ll_base_dataset_032918;
create multiset table dl_crm.mgua_ll_base_dataset_032918
as
(
select * from base 
) with data primary index (PSN,ACCOUNT_GUID)  
;
*/
/*
create multiset volatile table base as
(select * from dl_crm.mgua_ncore_base_dataset_032918)
with data on commit preserve rows ;*/
--------------------------------------------------------------------------------------------
---------------------- FEATURE APPEND
--------------------------------------------------------------------------------------------


----------------------------------------------
/*
 Email Touchpoint and Engagement
 CAMPAIGN_ID: (1138982,1164462,1199142,1216002,1223682,1220302)
 */
----------------------------------------------

create multiset volatile table temp_em
as(
select bs.*,
	   e.NUM_SENT,
	   e.NUM_OPENED,
	   e.NUM_CLICKED
from base bs
LEFT JOIN
(
select  a.CUST_ID,
		a.NUM_SENT,
		b.NUM_OPENED,
		c.NUM_CLICKED
from
(select CUST_ID,count(distinct CAMPN_ID) as NUM_SENT from
 CDCBVDB.BV_RSPNSYS_CAMPN_EV
 where 
 CAMPN_ID in (1138982,1164462,1199142,1216002,1223682,1220302)
 and EV_TYPE_ID = 1
 group by CUST_ID)a
 left join
 (select CUST_ID,count(distinct CAMPN_ID) as NUM_OPENED from
 CDCBVDB.BV_RSPNSYS_CAMPN_EV
 where
 CAMPN_ID in (1138982,1164462,1199142,1216002,1223682,1220302)
 and EV_TYPE_ID = 4
 group by CUST_ID)b on a.CUST_ID = b.CUST_ID
 left join
 (select CUST_ID,count(distinct CAMPN_ID) as NUM_CLICKED from
 CDCBVDB.BV_RSPNSYS_CAMPN_EV
 where
 CAMPN_ID in (1138982,1164462,1199142,1216002,1223682,1220302)
 and EV_TYPE_ID = 5
 group by CUST_ID)c on a.CUST_ID = c.CUST_ID) e
 on cast(e.CUST_ID as decimal(36,0)) = bs.ACCOUNT_GUID
 ) with data primary index (PSN,ACCOUNT_GUID) on commit preserve rows;


select count(*) from temp_em; --7909118






--- select count(*) from temp_em where NUM_SENT is not NULL;
----------------------------------------------
/*
 PIF Touchpoint and Engagement
 CAMPAIGN_ID: ('16719','17601','17729','17879')
 */
----------------------------------------------

------------------------------------------------------------------------------------------- 
-------- Create a table with pif campaign id's
------------------------------------------------------------------------------------------- 
---drop table pifcamid;
CREATE MULTISET VOLATILE TABLE pifcamid
    AS
    (
    SELECT
        PIF_CAMPN_ID,
        PIF_CAMPN_NM_ABBREV,
        PIF_CAMPN_PATH_NM,
        IS_CNTL_PATH,
        PIF_MSG_KEY,
        PIF_MSG_ID
        FROM CDCBVDB.BV_PIF_CAMPN
        		
        INNER JOIN
        CDCBVDB.BV_PIF_CAMPN_PATH
        ON BV_PIF_CAMPN_PATH.PIF_CAMPN_KEY = BV_PIF_CAMPN.PIF_CAMPN_KEY
        
        INNER JOIN
        CDCBVDB.BV_PIF_MSG
        ON BV_PIF_CAMPN_PATH.PIF_CAMPN_PATH_KEY = BV_PIF_MSG.PIF_CAMPN_PATH_KEY
        WHERE PIF_CAMPN_ID IN ('16719','17601','17729','17879')
    ) WITH DATA PRIMARY INDEX (PIF_MSG_KEY) ON COMMIT PRESERVE ROWS  
    ;
    
------------------------------------------------------------------------------------------- 
-------- For the above campaigns associate pif,all displays, clicks
-------  Add index to the pif displays
------------------------------------------------------------------------------------------- 
---drop table pif_event;


CREATE MULTISET VOLATILE TABLE pif_event
    AS
    (
    SELECT
        CAST(PIF_EV_DTTM AS DATE)  PIF_EV_DTTM,
        PIF_CAMPN_ID,
        PIF_CAMPN_NM_ABBREV,
        PIF_CAMPN_PATH_NM,
        IS_CNTL_PATH,
        IS_DSPL_FL,
        IS_REMIND_LATER_CLK_FL,
        GOTO_URL_FL,
        REMIND_FL,
        CAMPN_OUTP_FL,
        ALERT_DISMSS_FL,
        MSG_CNTL_CLK_FL,
        BV_PIF_EV.PIF_MSG_KEY,
        PROD_SER_NUM,
        HBID
      --  ,ROW_NUMBER() OVER (PARTITION BY PROD_SER_NUM
      --  ORDER BY PIF_EV_DTTM DESC) AS  DISPLAY_IND
        FROM CDCBVDB.BV_PIF_EV 
        INNER JOIN 	
        pifcamid
        ON BV_PIF_EV.PIF_MSG_KEY = pifcamid.PIF_MSG_KEY
        WHERE 
        CAST( PIF_EV_DTTM AS DATE) >= '2017-06-08'
            AND  PROD_SER_NUM IS NOT NULL
            AND  IS_DSPL_FL = '1'
    
    ) WITH DATA PRIMARY INDEX (PROD_SER_NUM) ON COMMIT PRESERVE ROWS  
    ;


 ------------------------------------------------------------------------------------------- 
---- Create a table total displays, clicks, 
----               by PSN
------------------------------------------------------------------------------------------- 
drop table pif_event1;
CREATE MULTISET VOLATILE TABLE pif_event1
    AS
    (
    SELECT a.PROD_SER_NUM,
        TOTAL_DSPL_FL,
        TOTAL_CLK_A,
        TOTAL_CLK_B
        FROM
pif_event a
LEFT JOIN        
        (SELECT PROD_SER_NUM,count(distinct PIF_CAMPN_ID) as TOTAL_DSPL_FL
        from pif_event
        where IS_DSPL_FL = 1
        group by PROD_SER_NUM     
        ) d on a.PROD_SER_NUM = d.PROD_SER_NUM
LEFT JOIN        
        (SELECT PROD_SER_NUM,count(distinct PIF_CAMPN_ID) as TOTAL_CLK_A
        from pif_event
        where GOTO_URL_FL = 1
        group by PROD_SER_NUM     
        ) ca on a.PROD_SER_NUM = ca.PROD_SER_NUM
LEFT JOIN        
        (SELECT PROD_SER_NUM,count(distinct PIF_CAMPN_ID) as TOTAL_CLK_B
        from pif_event
        where MSG_CNTL_CLK_FL = 1
        group by PROD_SER_NUM     
        ) cb on a.PROD_SER_NUM = cb.PROD_SER_NUM        
    ) WITH DATA PRIMARY INDEX (PROD_SER_NUM) ON COMMIT PRESERVE ROWS ;
---select top 10 * from pif_event1 where TOTAL_CLK_A is not null;   
------------------------------------------------------------------------------------------- 
 ---- associate account_guid with these psn's
------------------------------------------------------------------------------------------- 

drop table temp_pf;
CREATE MULTISET VOLATILE TABLE temp_pf
    AS
    (
 SELECT DISTINCT
        a.*,
       --- a.order_number,
      ---  a.PROD_FMLY_GRP,
        TOTAL_DSPL_FL,
        TOTAL_CLK_A,
        TOTAL_CLK_B
 FROM  temp_em a
 LEFT JOIN
        pif_event1 ON pif_event1.PROD_SER_NUM = a.PSN
/*        
LEFT JOIN
        (
        SELECT
            ACCOUNT_GUID,
            MAX(LL_ACTIVE_FLG) LL_ACTIVE_FLG
            FROM DL_CRM.LL_MATCH2
            WHERE LL_ACTIVE_FLG = '1'
            GROUP BY
            ACCOUNT_GUID
        ) LL_MATCH2
        ON CAST(LL_MATCH2.ACCOUNT_GUID AS FLOAT) = a.ACCOUNT_GUID
 */   	
    ) WITH DATA PRIMARY INDEX (PSN, ACCOUNT_GUID) ON COMMIT PRESERVE ROWS
    ;    
   
 select count(*) from temp_pf;   ---7909118
 select TOTAL_CLK_A, avg(LL_FL) from temp_pf group by TOTAL_CLK_A;
 select * from temp_pf where TOTAL_CLK_A is not null;
 
 
 ----------------------------------------------
--- add tenure
----------------------------------------------
-- Start dates:
--- drop table base2;
CREATE MULTISET VOLATILE TABLE base2
AS
( select t.*,
		 COALESCE(a.ORDER_DATE,START_DATE) START_DATE,
		 case when LL_FL = 0 then CURRENT_DATE - START_DATE
		 	else t.ORDER_DATE - START_DATE end as TENURE_DAYS
FROM temp_pf t
 	LEFT JOIN
		(
  			SELECT temp_pf.ACCOUNT_GUID ,
         		MIN(b.START_DATE) AS START_DATE ,
         		MIN(b.ORDER_DATE) AS ORDER_DATE
  			FROM temp_pf
			left join CDCBVDB.BV_MT_REG_PSN_DTLS b
  			on temp_pf.ACCOUNT_GUID = b.ACCOUNT_GUID
  			GROUP BY
  			temp_pf.ACCOUNT_GUID) a 
  	on t.ACCOUNT_GUID = a.ACCOUNT_GUID
) WITH DATA PRIMARY INDEX (ACCOUNT_GUID) on commit preserve rows
;

select count(distinct(ACCOUNT_GUID)) from base2;----7909118





----------------------------------------------
--- add number of seats
--- seat usage
----------------------------------------------
-- Most recent product:

CREATE MULTISET VOLATILE TABLE base3
AS  
(
  SELECT
   a.*
  ,b.PROD_CNSMR
  ,b.PROD_FMLY_GRP as PROD_EXT
  ,b.PROD_VERS
  ,b.SEAT_CNT
  ,b.INIT_SUBS_PER
  ,b.SKU_TYPE
  ,b.MED_TYPE
  ,b.PROD_SUB_TYPE
  ,b.PTNR_NM
  ,b.PTNR_GRP
  ,(CASE
  WHEN b.INIT_SUBS_PER = 366 THEN '12 MONTHS'
  WHEN b.INIT_SUBS_PER = 732 THEN '24 MONTHS'
  WHEN b.INIT_SUBS_PER = 1098 THEN '36 MONTHS'
  WHEN b.INIT_SUBS_PER = 458 THEN '15 MONTHS'
  WHEN b.INIT_SUBS_PER = 31 THEN '01 MONTHS'
  WHEN b.INIT_SUBS_PER = 92 THEN '03 MONTHS'
  WHEN b.INIT_SUBS_PER = 1830 THEN '60 MONTHS'
  WHEN b.INIT_SUBS_PER = 2745 THEN '90 MONTHS'
  WHEN b.INIT_SUBS_PER = 183 THEN '06 MONTHS'
  ELSE 'OTHER'
  END
  ) as SUB_PERIOD
  FROM 
  base2 a 
  LEFT JOIN CDCBVDB.BV_ITEM_EXT b 
  ON COALESCE(a.SKUP, 'XXX') = b.SKU_NUM
) WITH DATA PRIMARY INDEX (ACCOUNT_GUID) on commit preserve rows
;

select count(*) from base3; --- 7909118






		
	
-----
---A: Active Opt In
---B: Active Opt Out
---P : Passive Opt In
---Q: Passive Opt Out
		
		
----------------------------------------------
--- number of active products
----------------------------------------------	
drop table base4;
CREATE MULTISET VOLATILE TABLE base4
AS  
(
select a.*,
       ACTIVE_PRODUCTS,
       LAST_INTERACTION_DATE
	from base3 a
		left join CDCBVDB.BV_MT_REG_ACCT_DTLS on a.ACCOUNT_GUID = BV_MT_REG_ACCT_DTLS.ACCOUNT_GUID	and BV_MT_REG_ACCT_DTLS.LAST_INTERACTION_DATE <= a.ORDER_DATE
	QUALIFY ROW_NUMBER () OVER (PARTITION BY a.ACCOUNT_GUID ORDER BY LAST_INTERACTION_DATE desc) = 1
	) WITH DATA PRIMARY INDEX (ACCOUNT_GUID) on commit preserve rows
	;
	
select count(*) from base4; ---7909118







----------------------------------------------
--- credit card, payment sub type
----------------------------------------------	
 ---- Executed as Single statement.  Failed [5315 : HY000] The user does not have SELECT access to DL_PR.bin_mapping.BIN. 

CREATE MULTISET VOLATILE TABLE cctemp1
AS
( 
  SELECT
  a.ACCOUNT_GUID
  ,b.WLT_LAST_UPDATED_DATE
  ,ADD_MONTHS(cast((SUBSTR(b.CARD_EXP_YEAR_MONTH,1,4) || LEAST(SUBSTR(b.CARD_EXP_YEAR_MONTH,5,2),12)) as date FORMAT 'YYYYMM'), 1) - 1 AS CARD_EXP_DATE
  ,b.BIN
  ,b.PAYMENT_TYPE_CODE
  ,b.PAYMENT_TYPE_NAME
  ,b.PAYMENT_SUBTYPE_CODE
  ,b.PAYMENT_SUBTYPE_NAME
  ,b.CITY
  ,b.STATE
  FROM base4 a
  LEFT JOIN CDCBVDB.BV_MT_WALLET b ON a.ACCOUNT_GUID = b.ACCOUNT_GUID
  where 
  CARD_EXP_DATE <= ORDER_DATE 
QUALIFY ROW_NUMBER () OVER (PARTITION BY a.ACCOUNT_GUID ORDER BY CARD_EXP_DATE DESC) = 1  
) WITH DATA PRIMARY INDEX (ACCOUNT_GUID) ON COMMIT PRESERVE ROWS
;
  

------ summarize the above table

CREATE MULTISET VOLATILE TABLE cctemp2
AS
(
  SELECT
  a.ACCOUNT_GUID
  ,a.CARD_EXP_DATE
  ,a.BIN
  ,a.PAYMENT_TYPE_CODE
  ,a.PAYMENT_TYPE_NAME
  ,a.PAYMENT_SUBTYPE_CODE
  ,a.PAYMENT_SUBTYPE_NAME
  ,a.CITY
  ,a.STATE
  ,b.CARD_BRAND
  ,b.ISSUING_ORG
  ,b.CARD_TYPE
  ,b.CARD_CATEGORY
  FROM cctemp1 a
  LEFT JOIN (
  SELECT
  BIN
  ,MAX(CARD_BRAND) AS CARD_BRAND
  ,MAX(ISSUING_ORG) AS ISSUING_ORG
  ,MAX(CARD_TYPE) AS CARD_TYPE
  ,MAX(CARD_CATEGORY) AS CARD_CATEGORY
  FROM DL_PR.bin_mapping
  WHERE
  BIN is not null
  GROUP BY
  BIN
  ) b ON a.BIN = b.BIN
) WITH DATA PRIMARY INDEX (ACCOUNT_GUID) on commit preserve rows
  ;   
  
	

---- join with the base4
CREATE MULTISET VOLATILE TABLE base5
AS
(
select distinct a.*,
	   			CARD_BRAND,
	   			CARD_TYPE,
	  		    CARD_CATEGORY
	from base4 a 
		left join cctemp2 b on a.ACCOUNT_GUID = b.ACCOUNT_GUID
		) WITH DATA PRIMARY INDEX (ACCOUNT_GUID) on commit preserve rows;

--- select count(*) from base5; ---7747457






----------------------------------------------
--- mac & windows usage
----------------------------------------------
drop table base6;
CREATE MULTISET VOLATILE TABLE base6
AS
(
select a.*,
       MAC_USAGE,
       Mobile_Apple_USAGE,
       Windows_USAGE,
       SEAT_USAGE
 from base5 a
 left join
(
  SELECT
  PSN
  ,SUM(MAC_IND) AS MAC_USAGE
  ,SUM(Mobile_Apple_IND) AS Mobile_Apple_USAGE
  ,SUM(Mobile_Android_IND) AS Mobile_Android_USAGE
  ,SUM(Windows_IND) AS Windows_USAGE
  ,Count(distinct coalesce(GUID,cast(FIRST_ACTIVATION_DATE as varchar(255)))) AS SEAT_USAGE
  FROM (
  SELECT
  a.PSN
  ,b.GUID
  ,b.FIRST_ACTIVATION_DATE
  ,CASE WHEN b.PRODUCT_NAME LIKE '%MAC%' THEN 1 ELSE 0 END AS MAC_IND
  ,CASE WHEN b.PRODUCT_NAME LIKE '%IOS%' THEN 1 ELSE 0 END AS Mobile_Apple_IND
  ,CASE WHEN b.PRODUCT_NAME LIKE ANY ('%Norton Mobile Security%','%NMS%') Then 1 ELSE 0 END AS Mobile_Android_IND
  ,CASE WHEN b.PRODUCT_NAME LIKE '%MAC%' THEN 0 WHEN b.PRODUCT_NAME LIKE '%IOS%' THEN 0 WHEN b.PRODUCT_NAME LIKE ANY ('%Norton Mobile Security%','%NMS%') Then 0   ELSE 1 END AS Windows_IND
  FROM base5 a
  LEFT JOIN CDCBVDB.BV_UNIT_CURRENT b
  ON a.PSN = b.PSN
  AND b.FINGERPRINT is not null
  AND (b.FIRST_ACTIVATION_DATE <= a.ORDER_DATE ) ---- want the activations before order date or current_date
  --- AND (b.FIRST_ACTIVATION_DATE < '2008-01-01' OR STATE = 1) why ?
  ) c
  GROUP BY
  PSN
  ) d
ON a.PSN = d.PSN
	
)
WITH DATA PRIMARY INDEX (PSN) on commit preserve rows
;

select count(*) from base6;      --- 7909118
select top 10 * from base5;
-----------------
--- Purchase CORE prior to LL
-----------------

drop table base7;
CREATE MULTISET VOLATILE TABLE base7
AS
(
select a.*,
      (CASE WHEN CORE_ORDER_DATE is NULL THEN 0
            ELSE 1
        END) CORE_FL,
    CORE_ORDER_DATE
from base6 a
LEFT JOIN
(
SELECT
ACCOUNT_GUID,
COALESCE (BV_MT_REG_PSN_DTLS.ORDER_DATE, START_DATE) NEW_ORDER_DATE,
(CASE
  WHEN ACQUISITION_CHANNEL = 'RETAIL' THEN ACTIVATION_DATE
  ELSE NEW_ORDER_DATE
  END
  ) as CORE_ORDER_DATE
FROM
CDCBVDB.BV_MT_REG_PSN_DTLS
INNER JOIN
CDCBVDB.BV_ITEM_EXT
ON BV_ITEM_EXT.SKU_NUM = BV_MT_REG_PSN_DTLS.SKUP 
WHERE
PAID_STATUS = 'PAID'
AND
ACQUISITION_CHANNEL <> 'CSP'
AND 
EXPIRATION_DATE >= CURRENT_DATE
AND 
PROD_FMLY_GRP IN ('NCORE')
AND
IS_CORE = 'Y'
AND
CORE_ORDER_DATE is not null
AND
ACCOUNT_GUID is not null

QUALIFY ROW_NUMBER () OVER (PARTITION BY ACCOUNT_GUID ORDER BY CORE_ORDER_DATE desc) = 1 
)c
on a.ACCOUNT_GUID = c.ACCOUNT_GUID 
)
WITH DATA PRIMARY INDEX (ACCOUNT_GUID,PSN) on commit preserve rows;

 --------------------------
 --- Add Email Opt Out Flag
 ---------------------------
---select top 10 * from CDCBVDB.BV_MT_REG_ACCT_DTLS;
drop table base8;
CREATE MULTISET VOLATILE TABLE base8
AS(
select a.*,
       b.EMAIL_OPT_OUT
from base7 a
LEFT JOIN
	CDCBVDB.BV_MT_REG_ACCT_DTLS b on a.ACCOUNT_GUID = b.ACCOUNT_GUID
----where b.LAST_INTERACTION_DATE < a.ORDER_DATE
QUALIFY ROW_NUMBER () OVER (PARTITION BY a.ACCOUNT_GUID ORDER BY b.LAST_INTERACTION_DATE desc) = 1 

) WITH DATA PRIMARY INDEX (ACCOUNT_GUID,PSN) on commit preserve rows;


select count(*) from base8; ---7909118
select top 10 * from base8; 

create multiset table dl_crm.mgua_ll_dataset_03302018
as
(
select * from base8 
) with data primary index (PSN,ACCOUNT_GUID) 

;
---------
drop table base8_2;

CREATE MULTISET  TABLE dl_crm.razzak_ll_dataset_v3_06042018
AS(
select a.*,
       b.age,
       b.Wealth_Predictor_ScorAN127,
       b.Probable_Ordering_Through_Internet_MD007,
       b.Probable_Purchasing_Life_Insurance_MD014,
       b.contribute_political_conservative_vw120,
       b.Contribute_Political_Liberal_VW121,
       b.Politics_Donor_Liberal_Likelihood_AN142,
       b.Politics_Donor_Conservative_Likelihood_AN141,
       b.Wealth_Profile2_AN128,
       b.Merkle_Adjusted_Wealth_Rating_AU003,
       b.age_bucket,
       b.occupation,
       b.gender,
       b.inferred_education_AU015,
       b.marital_status,
       b.Facebook_Influencer_AN131,
       b.Children_DS907,
       b.Homeowner_DS921,
       b.Home_Value_DS922,
       b.income,
       b.Number_Persons_Living_Unit_MS947,
       b.Geo_Latitude_MS514,
       b.Geo_Longitude_MS516
from base8 a
LEFT JOIN
	dl_nbu.ipa_demo_norton3 b on a.ACCOUNT_GUID = b.NORTONGUID
) WITH DATA PRIMARY INDEX (ACCOUNT_GUID,PSN);


on commit preserve rows;


select top 10 * from dl_crm.mgua_ll_dataset_03302018_demo;
--------


select count(*),EMAIL_OPT_OUT from dl_crm.mgua_ll_dataset_03302018 group by EMAIL_OPT_OUT; ---{N:4891672, Y:2548551, null:327792}


 
 
 
select count(*),NUM_SENT from dl_crm.mgua_ll_dataset_03302218 group by NUM_SENT;
select count(*),NUM_SENT,NUM_CLICKED,LL_FL from dl_crm.mgua_ll_dataset_03302218 group by NUM_SENT,NUM_CLICKED,LL_FL having LL_FL=1;
 
 --------------------------
 --- Add Email Opt Out Flag
 ---------------------------
 
 
 
 --------------------------
 -- Metrics 
 --------------------------
 select count(*) from dl_crm.mgua_ll_dataset_03302018 where NUM_SENT=1 ; 

 -----Data Check
 CREATE MULTISET VOLATILE TABLE tmp_em
AS (
select  a.CUST_ID,
		a.NUM_SENT,
		b.NUM_OPENED,
		c.NUM_CLICKED
from
(select CUST_ID,count(distinct CAMPN_ID) as NUM_SENT from
 CDCBVDB.BV_RSPNSYS_CAMPN_EV
 where
 CAMPN_ID in (1138982,1164462,1199142,1216002,1223682,1220302)
 ----(1138982,1164462,1199142,1216002,1028522,1008122,957382,977722,1223682,1220302)
 and EV_TYPE_ID = 1
 group by CUST_ID)a
 left join
 (select CUST_ID,count(distinct CAMPN_ID) as NUM_OPENED from
 CDCBVDB.BV_RSPNSYS_CAMPN_EV
 where
 CAMPN_ID in (1138982,1164462,1199142,1216002,1223682,1220302)
 and EV_TYPE_ID = 4
 group by CUST_ID)b on a.CUST_ID = b.CUST_ID
 left join
 (select CUST_ID,count(distinct CAMPN_ID) as NUM_CLICKED from
 CDCBVDB.BV_RSPNSYS_CAMPN_EV
 where
 CAMPN_ID in (1138982,1164462,1199142,1216002,1223682,1220302)
 and EV_TYPE_ID = 5
 group by CUST_ID)c on a.CUST_ID = c.CUST_ID) WITH DATA PRIMARY INDEX (CUST_ID) ON COMMIT PRESERVE ROWS;

 select count(*) from tmp_em where NUM_SENT = 1;

(1138982,1131482, 1138882,1164462,1199142,1216002)

select distinct CAMPN_NM,CAMPN_ID from CDCBVDB.BV_RSPNSYS_CAMPN where CAMPN_NM like '%LIFELOCK%';
select * from tmp_em where NUM_SENT = 1
and
cast(CUST_ID as decimal(36,0)) not in (select distinct ACCOUNT_GUID from temp_em where NUM_SENT = 1); 
select * from CDCBVDB.BV_MT_REG_PSN_DTLS where ACCOUNT_GUID = -640101376378043497594479439226637395;
select count(*) from DL_CRM.LL_MATCH2 where LL_ACTIVE_FLG = 1 where ACCOUNT_GUID = '-640101376378043497594479439226637395';

/*
select CUST_ID,count(*) as NUM_SENT from
 CDCBVDB.BV_RSPNSYS_CAMPN_EV
 where
 CAMPN_ID in (1138982,1131482, 1138882,1164462,1199142,1216002)
 and EV_TYPE_ID = 1
 group by CUST_ID;
 
*/
select top 10 * from cast(e.CUST_ID as decimal(36,0))M DL_CRM.LL_MATCH3
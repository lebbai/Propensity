
--for table 2
CREATE MULTISET VOLATILE TABLE omini_tmp1
AS (
select  INCM_PSN_NUM, EVR_41_PRP_41_SITE_SECTN_NM,VISIT_STRT_PG_URL_NM, PG_URL_NM,VISIT_STRT_DTTM_GMT, 
HIT_DTTM_GMT, USER_AGNT_DESC, REFRRER_URL_NM
FROM EDW_KATAMARI_V.OMNITURE_HITS where cast(VISIT_STRT_DTTM_GMT as DATE) >= '2017-03-01'  
AND cast(VISIT_STRT_DTTM_GMT as DATE) < CURRENT_DATE AND INCM_PSN_NUM IS NOT NULL 
GROUP BY 1,2,3,4,5,6,7,8
)WITH DATA PRIMARY INDEX (INCM_PSN_NUM) ON COMMIT PRESERVE ROWS;

select * from omini_tmp1






--fill out the referal page for each 
CREATE MULTISET VOLATILE TABLE omini_refer_page AS
(
SELECT 
INCM_PSN_NUM, 
VISIT_STRT_DTTM_GMT, 
max(REFRRER_URL_NM) as REFRRER_URL_NM , 
INCM_PSN_NUM ||'-'|| CAST(VISIT_STRT_DTTM_GMT as VARCHAR(50)) as Session_ID  
from omini_tmp1
group by INCM_PSN_NUM, VISIT_STRT_DTTM_GMT

)WITH DATA PRIMARY INDEX (INCM_PSN_NUM) ON COMMIT PRESERVE ROWS;

select count(*) from omini_refer_page
--select * from omini_refer_page

/*
CREATE MULTISET VOLATILE TABLE omini_psn AS
(SELECT INCM_PSN_NUM,
ROW_NUMBER() OVER (ORDER BY INCM_PSN_NUM DESC ) AS PSN_ROWNumber
FROM DL_CRM.razzak_Omniture_hits_for_Monter_carlo_aggragation GROUP BY 1

)WITH DATA PRIMARY INDEX (INCM_PSN_NUM) ON COMMIT PRESERVE ROWS;
*/

CREATE MULTISET VOLATILE TABLE omini_test
AS (
SELECT 
omi.INCM_PSN_NUM, 
omi.EVR_41_PRP_41_SITE_SECTN_NM,
omi.VISIT_STRT_PG_URL_NM, 
omi.PG_URL_NM,
omi.VISIT_STRT_DTTM_GMT, 
omi.HIT_DTTM_GMT, 
omi.USER_AGNT_DESC,
omi.INCM_PSN_NUM ||'-'|| CAST(omi.VISIT_STRT_DTTM_GMT as VARCHAR(50)) as Session_ID 
FROM  omini_tmp1 as omi

)WITH DATA PRIMARY INDEX (INCM_PSN_NUM) ON COMMIT PRESERVE ROWS;


select * from omini_test

--get the 1st row off each session with time stamp
--drop table omini_test_single
CREATE MULTISET VOLATILE TABLE omini_test_single
AS (
SELECT * from omini_test
QUALIFY ROW_NUMBER () OVER (PARTITION BY Session_ID ORDER BY HIT_DTTM_GMT DESC) = 1
)WITH DATA PRIMARY INDEX (INCM_PSN_NUM) ON COMMIT PRESERVE ROWS;



--get the 1st row off each session with time stamp
drop table omini_test_final

CREATE MULTISET VOLATILE TABLE omini_test_final AS
(
SELECT omi.INCM_PSN_NUM, omi.Session_ID, omi.VISIT_STRT_DTTM_GMT,  omi.USER_AGNT_DESC, refpg.REFRRER_URL_NM,

 				(CASE
  					WHEN omi.VISIT_STRT_PG_URL_NM IS NULL THEN omi.PG_URL_NM
  					ELSE omi.VISIT_STRT_PG_URL_NM
  					END
 				 ) as Landing_page,
 				 psd.ACCOUNT_GUID
 				 
 				 
FROM omini_test_single as omi
INNER JOIN omini_refer_page refpg
ON omi.Session_ID=refpg.Session_ID

LEFT JOIN 
(select PSN, ACCOUNT_GUID FROM CDCBVDB.BV_MT_REG_PSN_DTLS WHERE PSN IS NOT NULL AND  ACCOUNT_GUID IS NOT NULL GROUP BY PSN, ACCOUNT_GUID ) as psd
ON omi.INCM_PSN_NUM=psd.PSN



)WITH DATA PRIMARY INDEX (INCM_PSN_NUM) ON COMMIT PRESERVE ROWS;



select count(*) from omini_test_final where ACCOUNT_GUID IS  NULL



 --create a permanent table at DL_CRM
drop table DL_CRM.razzak_Omniture_hits_Monter_carlo_aggragation 

CREATE MULTISET TABLE DL_CRM.razzak_Omniture_hits_Monter_carlo_aggragation  AS
(
SELECT  * FROM omini_test_final


)WITH DATA PRIMARY INDEX (INCM_PSN_NUM);




select * from DL_CRM.razzak_Omniture_hits_Monter_carlo_aggragation 








create multiset volatile table referal_page
as( 
select INCM_PSN_NUM, Session_ID, VISIT_STRT_DTTM_GMT,ACCOUNT_GUID, REFRRER_URL_NM,
(CASE
  	WHEN max(REFRRER_URL_NM)  IS NULL THEN 'NONE'
  	ELSE count(REFRRER_URL_NM)
  		END
 	) as count_REFRRER_URL_NM
	
--MAX(REFRRER_URL_NM) over (partition by Session_ID ORDER BY VISIT_STRT_DTTM_GMT desc) AS INSD_GOVT_ID
from DL_CRM.razzak_Omniture_hits_Monter_carlo_aggragation  group by ACCOUNT_GUID, INCM_PSN_NUM, Session_ID,VISIT_STRT_DTTM_GMT,REFRRER_URL_NM
)WITH DATA PRIMARY INDEX (INCM_PSN_NUM) ON COMMIT PRESERVE ROWS;



--get number of landing page
create multiset volatile table temp_landing_page
as( 
select INCM_PSN_NUM, Session_ID, VISIT_STRT_DTTM_GMT,landing_page, count(landing_page) as count_landing_page

--MAX(REFRRER_URL_NM) over (partition by Session_ID ORDER BY VISIT_STRT_DTTM_GMT desc) AS INSD_GOVT_ID
from DL_CRM.razzak_Omniture_hits_Monter_carlo_aggragation  group by  INCM_PSN_NUM,Session_ID,VISIT_STRT_DTTM_GMT, landing_page
)WITH DATA PRIMARY INDEX (INCM_PSN_NUM) ON COMMIT PRESERVE ROWS;

--add landing page to base
create multiset volatile table base9 AS
(
SELECT base.*, lp.landing_page from dl_crm.razzak_ll_dataset_v3_06042018 base
LEFT JOIN 

(select INCM_PSN_NUM, VISIT_STRT_DTTM_GMT,landing_page  from temp_landing_page 
QUALIFY ROW_NUMBER () OVER (PARTITION BY INCM_PSN_NUM ORDER BY VISIT_STRT_DTTM_GMT  DESC ) = 1) AS lp
ON base.PSN = lp.INCM_PSN_NUM
)WITH DATA PRIMARY INDEX (PSN) ON COMMIT PRESERVE ROWS;



--add referal page to base
create multiset volatile table base10 AS
(
SELECT base.*, rp.REFRRER_URL_NM from base9  base
LEFT JOIN 
(select INCM_PSN_NUM, VISIT_STRT_DTTM_GMT,REFRRER_URL_NM  from referal_page
QUALIFY ROW_NUMBER () OVER (PARTITION BY INCM_PSN_NUM ORDER BY VISIT_STRT_DTTM_GMT  DESC ) = 1) AS rp
ON base.PSN = rp.INCM_PSN_NUM
)WITH DATA PRIMARY INDEX (PSN) ON COMMIT PRESERVE ROWS;




--add referal page to base
create multiset volatile table base11 AS
(
SELECT base.*, ns.number_session from base10  base
LEFT JOIN 
(select INCM_PSN_NUM, count(Session_ID) as number_session  from DL_CRM.razzak_Omniture_hits_Monter_carlo_aggragation group by INCM_PSN_NUM) as ns
ON base.PSN = ns.INCM_PSN_NUM
)WITH DATA PRIMARY INDEX (PSN) ON COMMIT PRESERVE ROWS;


CREATE MULTISET  TABLE dl_crm.razzak_ll_dataset_v3_06042018_omni
AS(
SELECT * FROM base11
)WITH DATA PRIMARY INDEX (ACCOUNT_GUID,PSN);

select * from dl_crm.razzak_ll_dataset_v3_06042018_omni
--get user agen for last visit
create multiset volatile table user_agent AS
(
SELECT base.PSN, base.ACCOUNT_GUID, ua.USER_AGNT_DESC from dl_crm.razzak_ll_dataset_v3_06042018_omni  base
LEFT JOIN
(select INCM_PSN_NUM, VISIT_STRT_DTTM_GMT, USER_AGNT_DESC  from DL_CRM.razzak_Omniture_hits_Monter_carlo_aggragation
QUALIFY ROW_NUMBER () OVER (PARTITION BY INCM_PSN_NUM ORDER BY VISIT_STRT_DTTM_GMT  DESC ) = 1) AS ua
ON base.PSN = ua.INCM_PSN_NUM
)WITH DATA PRIMARY INDEX (PSN) ON COMMIT PRESERVE ROWS;


select count(*) from dl_crm.razzak_ll_dataset_v3_06042018_omni


select count(*) from addwealth_predict_score



drop table addwealth_predict_score

CREATE MULTISET  volatile table addwealth_predict_score 
AS(
select a.*,
       
       b.Wealth_Predictor_ScorAN127
       
from dl_crm.razzak_ll_dataset_v3_06042018_omni a
LEFT JOIN
	dl_nbu.ipa_demo_norton3 b on a.ACCOUNT_GUID = b.NORTONGUID
) WITH DATA PRIMARY INDEX (ACCOUNT_GUID,PSN) ON COMMIT PRESERVE ROWS;




CREATE MULTISET   table dl_crm.razzak_ll_dataset_v3_06042018_omni_v4
AS(
select a.*,
       
       b.Credit_Score_Range_VW107,
       b.education,
       b.Wealth_Predictor_ScorAN127
       
from dl_crm.razzak_ll_dataset_v3_06042018_omni a
LEFT JOIN
	dl_nbu.ipa_demo_norton3 b on a.ACCOUNT_GUID = b.NORTONGUID
) WITH DATA PRIMARY INDEX (ACCOUNT_GUID,PSN);

drop table dl_crm.razzak_ll_dataset_v3_06042018_omni_v3

select * from dl_crm.razzak_ll_dataset_v3_06042018_omni_v4

select Credit_Score_Range_VW107 from dl_nbu.ipa_demo_norton3



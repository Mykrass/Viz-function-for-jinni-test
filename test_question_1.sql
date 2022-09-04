with user_data as (
 select candidate_id
	, min(hire_reported::date) over(partition by candidate_id) as created
    , hire_reported::date as hire_date
	, count(candidate_id) over () as total_users
 from jinni
),
	 
hiring_count as (
 select candidate_id
     , count(*) as hiring
     , max(total_users) as total_users
      -- , round(sum(priceusd)/count(distinct devtodevid), 2) as "ARPPU"
      --, to_char(sum(p.priceusd)/count(distinct devtodevid), '$999,999,999,999.00') as "ARPPU"
      --, '$'||round((sum(p.priceusd)/count(distinct devtodevid))::numeric,2)|| ' per paying user' as "ARPPU"
  from user_data
  where hire_date between '2020-01-01' and '2020-12-31'
  group by 1
  order by 2 desc
),
  
selected_group as (
  select *
  , count(*) over() as total_selected_group_users
  from hiring_count
  where hiring >=2
  ),
  
fin_table as (
  select 100*max(total_selected_group_users)::float /max(total_users) as "% of users"
  from selected_group
  )

select *
from fin_table

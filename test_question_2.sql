with user_data as (
 select candidate_id
	, min(hire_reported::date) over(partition by candidate_id) as created
    , hire_reported::date as hire_date
	, count(candidate_id) over () as total_users
	, hire_salary
 from jinni
),
	 
hiring_count as (
 select candidate_id
     , count(*) as hiring
     , max(total_users) as total_users
	 , max(hire_salary) as hire_salary
  from user_data
  where hire_date between '2020-01-01' and '2020-12-31'
  group by 1
  order by 2 desc
),
  
selected_group as (
  select hiring
	, avg(hire_salary) as averagy_salery
  from hiring_count
  group by 1
  )

select *
from selected_group

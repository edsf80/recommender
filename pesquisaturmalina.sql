SELECT i.id, count(ustc.user_story_id) FROM turmalina.issue i
   inner join turmalina.user_story_test_case ustc on ustc.user_story_id = i.id
   inner join turmalina.test_specification ts on ustc.test_case_id = ts.id
where i.issue_type = 'USER_STORY'
  and i.acceptance_criteria is not null
  and i.status != 'CANCELED'
  and i.status != 'REJECTED'
group by i.id
  

select ut.user_story_id, count(ut.test_case_id) from turmalina.test_specification ts
  inner join turmalina.user_story_test_case ut on ut.test_case_id = ts.id
group by ut.user_story_id

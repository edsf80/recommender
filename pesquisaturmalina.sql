SELECT distinct p.description, concat('#',i.id), i.title FROM turmalina.issue i
   inner join turmalina.project p on i.project_id = p.id
   inner join turmalina.user_story_test_case ustc on ustc.user_story_id = i.id
   inner join turmalina.test_specification ts on ustc.test_case_id = ts.id
where i.issue_type = 'USER_STORY'
  and i.acceptance_criteria is not null
  and i.acceptance_criteria <> ''
  and i.status != 'CANCELED'
  and i.status != 'REJECTED'
  and p.id = 5
group by p.description, i.id, i.acceptance_criteria

# CAs
set @rownum := 0;

SELECT distinct p.description, concat('#',i.id), @rownum := @rownum + 1 as rownumber, i.acceptance_criteria FROM turmalina.issue i
   inner join turmalina.project p on i.project_id = p.id
   inner join turmalina.user_story_test_case ustc on ustc.user_story_id = i.id
   inner join turmalina.test_specification ts on ustc.test_case_id = ts.id
where i.issue_type = 'USER_STORY'
  and i.acceptance_criteria is not null
  and i.acceptance_criteria <> ''
  and i.status != 'CANCELED'
  and i.status != 'REJECTED'
  and p.id = 8
group by p.description, i.id, i.acceptance_criteria

 
# TCs
select distinct p.description, concat('#',i.id), i.title, tsv.id, tsv.name from turmalina.test_specification_version tsv
 inner join turmalina.user_story_test_case ustc on ustc.test_case_id = tsv.specification_id
 inner join turmalina.issue i on ustc.user_story_id = i.id
 inner join turmalina.project p on i.project_id = p.id
where p.id = 3

# TCs


# decidi não incluir USs sem AC
select * from turmalina.issue where id = 1821

select distinct p.id, p.description from turmalina.project p
  inner join turmalina.issue i on p.id = i.project_id
  inner join turmalina.user_story_test_case ut on ut.user_story_id = i.id
  inner join turmalina.test_specification ts on ut.test_case_id = ts.id
  
select ts.*, ut.user_story_id from turmalina.test_specification_version ts
   inner join turmalina.user_story_test_case ut on ts.id = ut.test_case_id   
where ut.user_story_id = 1809

# TCs - Buscando pelo requisito
select distinct p.id, p.description, concat('#',i.id), i.status, i.title, i.acceptance_criteria, ts.id, tsv.name  from turmalina.test_specification ts 
 inner join turmalina.requirement r on ts.requirement_id = r.id
 inner join turmalina.project p on r.project_id = p.id
 inner join turmalina.test_specification_version tsv on ts.id = tsv.specification_id
 inner join turmalina.issue i on r.id = i.requirement_id 
where p.id = 5

# USs - Buscando pelo requisito
select distinct p.description, concat('#',i.id) as id_us, i.title, i.acceptance_criteria, tsv.name, tsv.precondition, tsv.purpose from turmalina.issue i
  inner join turmalina.requirement r on i.requirement_id = r.id
  inner join turmalina.test_specification ts on ts.requirement_id = r.id
  inner join turmalina.test_specification_version tsv on ts.id = tsv.specification_id
  inner join turmalina.project p on i.project_id = p.id
where i.issue_type = 'USER_STORY'
  and i.acceptance_criteria is not null
  and i.acceptance_criteria <> ''
  and i.status != 'CANCELED'
  and i.status != 'REJECTED'  
  and tsv.closed = 1
  and i.project_id = 80
  and i.id = 997
  
 select distinct p.id, p.description from turmalina.project p
  inner join turmalina.issue i on p.id = i.project_id
  inner join turmalina.requirement r on i.requirement_id = r.id  
  inner join turmalina.test_specification ts on r.id = ts.requirement_id


 inner join turmalina.test_specification_version tsv on ts.id = tsv.specification_id 
 inner join turmalina.user_story_test_case ustc on ustc.test_case_id = ts.id
 
 inner join turmalina.issue i on ustc.user_story_id = i.id
 
 
 
 inner join turmalina.project p on i.project_id = p.id
where p.id = 3

select * from turmalina.test_specification_version tsv
inner join turmalina.test_specification ts on ts.id = tsv.specification_id
inner join turmalina.user_story_test_case ut on ut.test_case_id = ts.id
inner join turmalina.requirement r on r.id = ut.requirement_id
inner join turmalina.requirement_version rv on rv.requirement_id = r.id
where tsv.id = 3219
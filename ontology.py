from owlready2 import *

onto = get_ontology('https://test.org/onto.owl')

with onto:
    class part_of(Thing >> Thing, TransitiveProperty):
        """
        The "part-of" relation describes the hyponym-hypernym relation. The transitive property ensures that
        if a~b and b~c then a~c.
        """
        pass
    
    class JobPosting(Thing):
        """
        The source of information pertaining to labour market demand. Each posted job advert contains a jobtitle and 
        the job description.
        """
        pass
    
    class JobTitle(Thing):
        """
        The name of the occupation that the job description describes.
        """
        pass
    
    class WorkerQuality(Thing):
        """
        A knowledge, skill, ability or competence.
        """
        pass
    
    class Knowledge(WorkerQuality):
        pass
    
    class Skill(WorkerQuality):
        pass
    
    class Ability(WorkerQuality):
        pass
    
    class Competence(WorkerQuality):
        pass
    
    AllDisjoint([Knowledge, Skill, Ability, Competence])
    
    class described_by(JobTitle >> JobPosting):
        pass
    
    class describes(JobPosting >> JobTitle):
        inverse_property = described_by

    class requires_worker_qualities(JobTitle >> WorkerQuality):
        pass
        
    class has_occurrence(DataProperty, FunctionalProperty):
        domain = [WorkerQuality]
        range = [int]
        
    class has_transversality(DataProperty, FunctionalProperty):
        domain = [WorkerQuality]
        range = [float]
    
    class Transversal(WorkerQuality):
        equivalent_to = [WorkerQuality & has_transversality.only(ConstrainedDatatype(float, min_inclusive = 0.5))]
        pass
    
    class Specific(WorkerQuality):
        equivalent_to = [WorkerQuality & has_transversality.only(ConstrainedDatatype(float, max_exclusive = 0.5))]
        pass
    
    AllDisjoint([Transversal, Specific])
    WorkerQuality.equivalent_to.append(Knowledge|Skill|Ability|Competence|Transversal|Specific)

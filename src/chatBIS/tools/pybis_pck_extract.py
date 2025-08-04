import json, itertools as it
KEPT_PREFIXES = (
    "pybis.pybis.Openbis.", "pybis.sample.Sample.", "pybis.dataset.DataSet.",
    "pybis.experiment.Experiment.", "pybis.project.Project.", "pybis.space.Space.",
    "pybis.material.Material.", "pybis.group.Group.", "pybis.person.Person.",
    "pybis.tag.Tag.", "pybis.entity_type.", "pybis.vocabulary.Vocabulary.",
    "pybis.vocabulary.VocabularyTerm.", "pybis.pybis.ExternalDMS.",
    "pybis.pybis.PersonalAccessToken.", "pybis.role_assignment.RoleAssignment."
)

with open("pybis_api.json") as fp:
    api = json.load(fp)

slim_api = {
    k: v for k, v in api.items()
    if k.startswith(KEPT_PREFIXES)   # tuple works as multi-prefix test
}

with open("pybis_api_slim.json", "w") as fp:
    json.dump(slim_api, fp, indent=2)
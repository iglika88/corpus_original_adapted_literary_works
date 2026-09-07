# Metadata dictionary

The metadata fields and their predefined values are described below. A dash indicates that a field is not applicable or that the relevant information is unavailable.

## `id`

Identifier used in the automated corpus-construction pipeline.

## `lang`

Language of the current work. The predefined values are:

- *`en`: English*
- *`bg`: Bulgarian*
- *`el`: Greek*
- *`es`: Spanish*
- *`fr`: French*
- *`it`: Italian*
- *`ja`: Japanese*
- *`ru`: Russian*

## `title_en`

Common English title of the work.

## `title_here`

Title of the current work in the language in which it appears.

## `version`

Whether the current work is original or adapted. The predefined values are:

- *`original`: full, unabridged work; note that it may be a translation*
- *`adapted`: work that has been significantly altered, such as to increase its accessibility to a particular audience*

## `adp_aud`

Intended audience of the adaptation. The predefined values are:

- *`general`: no defined audience*
- *`children`: a younger audience*
- *`FL` (foreign language): learners of the language of the adapted work as a foreign language*
- *`LD` (learning difficulties): readers with learning difficulties or who otherwise need or prefer a related simplification intervention*
- *`theatre`: the adaptation is intended to be performed in front of an audience*

## `adp_subaud`

Intended subaudience of the adaptation, where applicable. Its interpretation depends on the value of `adp_aud`:

- *`children`*
  - *`{age}`: lower age limit proposed by the publisher or, alternatively, by a library or another established repository. It may be converted from a student grade level in the relevant country.*

- *`FL`*
  - *`{CEFR level}`: proficiency level proposed by the publisher or, alternatively, by a library or another established repository. It may be converted from another proficiency framework or from a student grade level in the relevant country.*

## `author_org`

Author of the original work.

## `author_here`

Author of the adaptation or translator of the current work, where applicable.

## `genre_org`

Genre of the original work. The predefined values are:

- *`novel`: may include novellas of substantial length*
- *`short story`: may include novellas of shorter length*
- *`fairy tale`*
- *`collection of tales`*
- *`epic poem`*
- *`parable`*
- *`myth`*
- *`fable`*
- *`play`*

## `year_org`

Year of first publication or, alternatively, composition of the original work.

## `year_here`

Year of publication of the current adaptation or translation, where applicable.

## `publisher`

Publisher and, where applicable, series of the current work.

## `diff_lang`

Whether the original work and the current work are in different languages.

## `adp_is_transl`

Whether the adaptation is itself a direct translation of an adaptation in another language.

## `nb_tokens`

Number of tokens in the current work, expressed in thousands (`K`) and rounded to two decimal places.

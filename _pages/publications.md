---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

You can also find my articles on [my Google Scholar profile](https://scholar.google.com/citations?user=zO1h0pQAAAAJ&hl=en).

{% include base_path %}

{% assign sorted_pubs = site.publications | sort: 'date' | reverse %}
{% assign pub_count = sorted_pubs | size %}

<ol reversed>
{% for post in sorted_pubs %}
  <li>
    <a href="{{ post.url }}"><strong>{{ post.title }}</strong></a><br>
    <em>{{ post.venue }}</em>, {{ post.date | date: "%Y" }}
    {% if post.paperurl %}
    [<a href="{{ post.paperurl }}">Paper</a>]
    {% endif %}
  </li>
{% endfor %}
</ol>

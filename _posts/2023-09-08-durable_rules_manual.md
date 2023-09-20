---
layout: default
title: 04. Durable Ruls
subtitle: 산업인공지능 개론 과목
---
-----

[PINBlog Gitea Repository](https://gitea.pinblog.codes/CBNU/04_durable_rules)

-----

# 04. Durable Rules로 규칙 구현하기
- 산업인공지능학과 대학원
    2022254026
        김홍열

### Durable Rules 란?
* 비즈니스 룰 엔진을 구현하기 위한 라이브러리로 python, ruby, nodejs로 구현할 수 있다.

### Durable Rules 설치하기 (Python)

``` planetext

pip install durable_rules

```
### Durable Rules Git
[github - jruizgit/rules](https://github.com/jruizgit/rules)


<details>
<summary>Durable Rules Manual</summary>
<div markdown="1">

##### 패키지 불러오기

``` planetext

from durable.lang import *

```

# Basic

<details>
<summary>Rules (Trigger: @when_all(==, &, <<))</summary>
<div markdown="1">

* 규칙은 프레임워크의 기본 구성 요소입니다.
* 규칙의 선행 조건은 규칙의 결과 조건(동작)을 실행하기 위해 충족되어야 하는 조건을 정의합니다.
* 관례적으로 m은 주어진 규칙에 의해 평가될 데이터를 나타냅니다.

``` python

with ruleset('test'):
    # antecedent
    @when_all(m.subject == 'World')
    def say_hello(c):
        # consequent
        print ('Hello {0}'.format(c.m.subject))
post('test', { 'subject': 'World' })

```

</div>
</details>

<details>
<summary>Facts (assert_fact)</summary>
<div markdown="1">

* 사실은 지식 기반을 정의하는 데이터를 나타냅니다.
* 사실은 JSON 객체로 주장되며, 취소될 때까지 저장됩니다.
* 사실이 규칙의 선행 조건을 만족하면, 규칙의 결과 조건이 실행됩니다.

``` python

with ruleset('animal'):
    # will be triggered by 'Kermit eats flies'
    @when_all((m.predicate == 'eats') & (m.object == 'flies'))
    def frog(c):
        c.assert_fact({ 'subject': c.m.subject, 'predicate': 'is', 'object': 'frog' })

    @when_all((m.predicate == 'eats') & (m.object == 'worms'))
    def bird(c):
        c.assert_fact({ 'subject': c.m.subject, 'predicate': 'is', 'object': 'bird' })

    # will be chained after asserting 'Kermit is frog'
    @when_all((m.predicate == 'is') & (m.object == 'frog'))
    def green(c):
        c.assert_fact({ 'subject': c.m.subject, 'predicate': 'is', 'object': 'green' })

    @when_all((m.predicate == 'is') & (m.object == 'bird'))
    def black(c):
        c.assert_fact({ 'subject': c.m.subject, 'predicate': 'is', 'object': 'black' })

    @when_all(+m.subject)
    def output(c):
        print('Fact: {0} {1} {2}'.format(c.m.subject, c.m.predicate, c.m.object))
assert_fact('animal', { 'subject': 'Kermit', 'predicate': 'eats', 'object': 'flies' })

```

</div>
</details>

<details>
<summary>Events (post)</summary>
<div markdown="1">

* 이벤트는 규칙에 전달되어 평가될 수 있습니다. 
* 이벤트란 일시적인 사실로, 결과를 실행하기 직전에 취소되는 사실입니다.
* 따라서 이벤트는 한 번만 관찰할 수 있습니다.
* 이벤트는 관찰될 때까지 저장됩니다.

``` python

with ruleset('risk'):
    @when_all(c.first << m.t == 'purchase',
              c.second << m.location != c.first.location)
    # the event pair will only be observed once
    def fraud(c):
        print('Fraud detected -> {0}, {1}'.format(c.first.location, c.second.location))
post('risk', {'t': 'purchase', 'location': 'US'})
post('risk', {'t': 'purchase', 'location': 'CA'})

```

##### ✨위 예제에서 Event가 아닌 Fact를 적용하면 다음과 같이 출력됩니다.

``` python

assert_fact('risk', {'t': 'purchase', 'location': 'US', 'last_location': None})
assert_fact('risk', {'t': 'purchase', 'location': 'CA', 'last_location': None})

```

``` plaintext

Fraud detected -> US, CA
Fraud detected -> CA, US

```
예에서 두 가지 사실 모두 첫 번째 조건인 m.t == 'purchase'를 충족하며, 각 사실은 첫 번째 조건을 충족한 사실과 관련하여 두 번째 조건인 m.location != c.first.location을 충족합니다.

이벤트는 일시적인 사실입니다. 사실이 발송될 예정이라면 즉시 취소됩니다. 위 예제에서 post를 사용할 때, 두 번째 쌍이 계산되는 시점에 이미 이벤트가 취소되어 있습니다.

발송 전에 이벤트를 취소함으로써 작업 실행 중 계산해야 할 조합의 수를 줄일 수 있습니다.

</div>
</details>

<details>
<summary>State (s, update_state)</summary>
<div markdown="1">

* 규칙의 결과가 실행될 때 컨텍스트 상태를 사용할 수 있습니다. 
* 동일한 컨텍스트 상태는 규칙 실행 간에 전달됩니다. 
* 컨텍스트 상태는 삭제될 때까지 저장됩니다. 
* 컨텍스트 상태 변경은 규칙에 의해 평가될 수 있습니다. 
* 관례적으로 s는 규칙에 의해 평가되는 상태를 나타냅니다.

``` python

with ruleset('flow'):
    # state condition uses 's'
    @when_all(s.status == 'start')
    def start(c):
        # state update on 's'
        c.s.status = 'next' 
        print('start')

    @when_all(s.status == 'next')
    def next(c):
        c.s.status = 'last' 
        print('next')

    @when_all(s.status == 'last')
    def last(c):
        c.s.status = 'end' 
        print('last')
        # deletes state at the end
        c.delete_state()
update_state('flow', { 'status': 'start' })

```

</div>
</details>

<details>
<summary>Identity (+속성, none(+속성))</summary>
<div markdown="1">

* 같은 속성 이름과 값이 있는 팩트들은 단언(asserted)되거나 철회(retracted)될 때 동등하다고 간주됩니다.
* 같은 속성 이름과 값이 있는 이벤트들은 게시 시간이 중요하기 때문에 게시될 때 서로 다른 것으로 간주됩니다.

``` python

with ruleset('bookstore'):
    # this rule will trigger for events with status
    @when_all(+m.status)
    def event(c):
        print('bookstore-> Reference {0} status {1}'.format(c.m.reference, c.m.status))

    @when_all(+m.name)
    def fact(c):
        print('bookstore-> Added {0}'.format(c.m.name))
        
    # this rule will be triggered when the fact is retracted
    @when_all(none(+m.name))
    def empty(c):
        print('bookstore-> No books')
```

``` python

# 단언(assert_fact)이 성공했기 때문에 예외를 발생시키지 않습니다. 
assert_fact('bookstore', {
    'name': 'The new book',
    'seller': 'bookstore',
    'reference': '75323',
    'price': 500
})

# 이미 단언(assert_fact)된 사실이기 때문에 MessageObservedError가 발생합니다. 
try:
    assert_fact('bookstore', {
        'reference': '75323',
        'name': 'The new book',
        'price': 500,
        'seller': 'bookstore'
    })
except BaseException as e:
    print('bookstore expected {0}'.format(e.message))

# 새로운 이벤트가 게시되기 때문에 예외를 발생시키지 않습니다. 
post('bookstore', {
    'reference': '75323',
    'status': 'Active'
})

# 새로운 이벤트가 게시되기 때문에 예외를 발생시키지 않습니다.
post('bookstore', {
    'reference': '75323',
    'status': 'Active'
})

retract_fact('bookstore', {
    'reference': '75323',
    'name': 'The new book',
    'price': 500,
    'seller': 'bookstore'
})

```

</div>
</details>

<details>
<summary>Correlated Sequence</summary>
<div markdown="1">

* 규칙은 서로 관련된 이벤트 또는 사실의 시퀀스를 효율적으로 평가하는 데 사용할 수 있습니다. 아래 예시의 사기 탐지 규칙은 세 가지 이벤트 패턴을 보여줍니다: 두 번째 이벤트 금액이 첫 번째 이벤트 금액의 200%를 초과하고 세 번째 이벤트 금액이 다른 두 이벤트의 평균보다 큽니다.
* 기본적으로 관련된 시퀀스는 서로 다른 메시지를 캡처합니다. 아래 예시에서 두 번째 이벤트는 두 번째와 세 번째 조건을 모두 만족하지만, 이벤트는 두 번째 조건에 대해서만 캡처됩니다. distinct 속성을 사용하여 서로 다른 이벤트 또는 사실의 상관 관계를 비활성화할 수 있습니다.
* when_all 주석은 이벤트 또는 사실의 시퀀스를 표현합니다. << 연산자는 이후 표현식에서 참조할 수 있는 이벤트 또는 사실의 이름을 지정하는 데 사용됩니다. 이벤트 또는 사실을 참조할 때 모든 속성을 사용할 수 있습니다. 산술 연산자를 사용하여 복잡한 패턴을 표현할 수 있습니다.
* 산술 연산자: +, -, *, /

``` python

from durable.lang import *

with ruleset('risk'):
    @when_all(# distinct(True),
              c.first << m.amount > 10,
              c.second << m.amount > c.first.amount * 2,
              c.third << m.amount > (c.first.amount + c.second.amount) / 2)
    def detected(c):
        print('fraud detected -> {0}'.format(c.first.amount))
        print('               -> {0}'.format(c.second.amount))
        print('               -> {0}'.format(c.third.amount))
        
post('risk', { 'amount': 50 })
post('risk', { 'amount': 200 })
post('risk', { 'amount': 251 })

```

</div>
</details>

<details>
<summary>Choice of Sequences</summary>
<div markdown="1">

* durable_rules는 보다 풍부한 이벤트 시퀀스를 표현하고 효율적으로 평가할 수 있게 해줍니다. 아래 예시에서 두 이벤트\사실 시퀀스 각각이 동작을 실행합니다.

* 다음 두 함수는 더 풍부한 이벤트 시퀀스를 정의하는 데 사용되고 결합할 수 있습니다:

all: 이벤트 또는 사실 패턴의 집합입니다. 동작을 실행하려면 모든 패턴이 일치해야 합니다.

any: 이벤트 또는 사실 패턴의 집합입니다. 어느 하나만 일치해도 동작이 실행됩니다.

``` python

from durable.lang import *

with ruleset('expense'):
    @when_any(all(c.first << m.subject == 'approve', 
                  c.second << m.amount == 1000), 
              all(c.third << m.subject == 'jumbo', 
                  c.fourth << m.amount == 10000))
    def action(c):
        if c.first:
            print ('Approved {0} {1}'.format(c.first.subject, c.second.amount))
        else:
            print ('Approved {0} {1}'.format(c.third.subject, c.fourth.amount))
    

post('expense', { 'subject': 'approve' })
post('expense', { 'amount': 1000 })
post('expense', { 'subject': 'jumbo' })
post('expense', { 'amount': 10000 })

```

</div>
</details>

<details>
<summary>Lack of Information</summary>
<div markdown="1">

* 일부 경우에는 정보 부족이 중요한 의미를 가집니다. none 함수는 관련된 시퀀스가 있는 규칙에서 정보 부족을 평가하는 데 사용할 수 있습니다.

* 참고: none 함수는 정보 부족에 대한 추론을 위해 정보가 필요합니다. 즉, 해당 규칙에 이벤트나 사실이 등록되지 않은 경우에는 동작을 실행하지 않습니다.

``` python

from durable.lang import *

with ruleset('risk'):
    @when_all(c.first << m.t == 'deposit',
              none(m.t == 'balance'),
              c.third << m.t == 'withdrawal',
              c.fourth << m.t == 'chargeback')
    def detected(c):
        print('fraud detected {0} {1} {2}'.format(c.first.t, c.third.t, c.fourth.t))
        
assert_fact('risk', { 't': 'deposit' })
assert_fact('risk', { 't': 'withdrawal' })
assert_fact('risk', { 't': 'chargeback' })

assert_fact('risk', { 'sid': 1, 't': 'balance' })
assert_fact('risk', { 'sid': 1, 't': 'deposit' })
assert_fact('risk', { 'sid': 1, 't': 'withdrawal' })
assert_fact('risk', { 'sid': 1, 't': 'chargeback' })
retract_fact('risk', { 'sid': 1, 't': 'balance' })

```

</div>
</details>

<details>
<summary>Nested Objects</summary>
<div markdown="1">

* 중첩된 이벤트 또는 사실에 대한 질의도 지원됩니다.
* . 표기법은 중첩된 객체의 속성에 대한 조건을 정의하는 데 사용됩니다.

``` python

from durable.lang import *

with ruleset('expense'):
    # use the '.' notation to match properties in nested objects
    @when_all(c.bill << (m.t == 'bill') & (m.invoice.amount > 50),
              c.account << (m.t == 'account') & (m.payment.invoice.amount == c.bill.invoice.amount))
    def approved(c):
        print ('bill amount  ->{0}'.format(c.bill.invoice.amount))
        print ('account payment amount ->{0}'.format(c.account.payment.invoice.amount))

```       

``` python

# one level of nesting
post('expense', {'t': 'bill', 'invoice': {'amount': 100}})

# two levels of nesting
post('expense', {'t': 'account', 'payment': {'invoice': {'amount': 100}}})

```

</div>
</details>

<details>
<summary>Arrays</summary>
<div markdown="1">

``` python

from durable.lang import *

with ruleset('risk'):
    # matching primitive array
    @when_all(m.payments.allItems((item > 100) & (item < 500)))
    def rule1(c):
        print('fraud 1 detected {0}'.format(c.m.payments))

    # matching object array
    @when_all(m.payments.allItems((item.amount < 250) | (item.amount >= 300)))
    def rule2(c):
        print('fraud 2 detected {0}'.format(c.m.payments))

    # pattern matching string array
    @when_all(m.cards.anyItem(item.matches('three.*')))
    def rule3(c):
        print('fraud 3 detected {0}'.format(c.m.cards))

    # matching nested arrays
    @when_all(m.payments.anyItem(item.allItems(item < 100)))
    def rule4(c):
        print('fraud 4 detected {0}'.format(c.m.payments))
        
post('risk', {'payments': [ 150, 300, 450 ]})
post('risk', {'payments': [ { 'amount' : 200 }, { 'amount' : 300 }, { 'amount' : 450 } ]})
post('risk', {'cards': [ 'one card', 'two cards', 'three cards' ]})
post('risk', {'payments': [ [ 10, 20, 30 ], [ 30, 40, 50 ], [ 10, 20 ] ]}) 

```

</div>
</details>

<details>
<summary>Facts and Events as rvalues</summary>
<div markdown="1">

* 스칼라 값(문자열, 숫자 및 부울 값) 외에도 표현식의 오른쪽에서 관찰된 사실이나 이벤트를 사용할 수 있습니다.

``` python

from durable.lang import *

with ruleset('risk'):
    # compares properties in the same event, this expression is evaluated in the client 
    @when_all(m.debit > m.credit * 2)
    def fraud_1(c):
        print('debit {0} more than twice the credit {1}'.format(c.m.debit, c.m.credit))

    # compares two correlated events, this expression is evaluated in the backend
    @when_all(c.first << m.amount > 100,
              c.second << m.amount > c.first.amount + m.amount / 2)
    def fraud_2(c):
        print('fraud detected ->{0}'.format(c.first.amount))
        print('fraud detected ->{0}'.format(c.second.amount))
        
post('risk', { 'debit': 220, 'credit': 100 })
post('risk', { 'debit': 150, 'credit': 100 })
post('risk', { 'amount': 200 })
post('risk', { 'amount': 500 })

```

</div>
</details>

# Consequents

<details>
<summary>Conflict Resolution</summary>
<div markdown="1">

* 이벤트와 사실 평가는 여러 결과를 초래할 수 있습니다. pri (중요도) 함수를 사용하여 트리거 순서를 제어할 수 있습니다. 낮은 값의 작업이 먼저 실행됩니다. 모든 작업의 기본값은 0입니다.

* 이 예시에서, 마지막 규칙이 가장 높은 우선순위를 가지고 있으므로 먼저 트리거됩니다.

``` python

from durable.lang import *

with ruleset('attributes'):
    @when_all(pri(3), m.amount < 300)
    def first_detect(c):
        print('attributes P3 ->{0}'.format(c.m.amount))
        
    @when_all(pri(2), m.amount < 200)
    def second_detect(c):
        print('attributes P2 ->{0}'.format(c.m.amount))
        
    @when_all(pri(1), m.amount < 100)
    def third_detect(c):
        print('attributes P1 ->{0}'.format(c.m.amount))
                
assert_fact('attributes', { 'amount': 50 })
assert_fact('attributes', { 'amount': 150 })
assert_fact('attributes', { 'amount': 250 })

```

</div>
</details>

<details>
<summary>Action Batches</summary>
<div markdown="1">

* 많은 수의 이벤트 또는 사실이 결과를 만족시킬 때, 결과는 일괄적으로 전달될 수 있습니다.

count: 동작을 예약하기 전에 규칙이 만족해야 하는 정확한 횟수를 정의합니다.

cap: 동작을 예약하기 전에 규칙이 만족해야 하는 최대 횟수를 정의합니다.

* 이 예시는 정확히 세 개의 승인을 일괄 처리하고 거절 수를 두 개로 제한합니다:

``` python

from durable.lang import *

with ruleset('expense'):
    # this rule will trigger as soon as three events match the condition
    @when_all(count(3), m.amount < 100)
    def approve(c):
        print('approved {0}'.format(c.m))

    # this rule will be triggered when 'expense' is asserted batching at most two results       
    @when_all(cap(2),
              c.expense << m.amount >= 100,
              c.approval << m.review == True)
    def reject(c):
        print('rejected {0}'.format(c.m))

post_batch('expense', [{ 'amount': 10 },
                                    { 'amount': 20 },
                                    { 'amount': 100 },
                                    { 'amount': 30 },
                                    { 'amount': 200 },
                                    { 'amount': 400 }])
assert_fact('expense', { 'review': True })

```

</div>
</details>

<details>
<summary>Async Actions</summary>
<div markdown="1">

* 결과 동작은 비동기적일 수 있습니다. 
* 동작이 완료되면 완료(complete) 함수를 호출해야 합니다. 
* 기본적으로 동작은 5초 후에 포기된 것으로 간주됩니다. 
* 이 값은 작업 함수에서 다른 숫자를 반환하거나 renew_action_lease를 호출함으로써 변경할 수 있습니다.

``` python

from durable.lang import *
import threading

with ruleset('flow'):
    timer = None

    def start_timer(time, callback):
        timer = threading.Timer(time, callback)
        timer.daemon = True    
        timer.start()

    @when_all(s.state == 'first')
    # async actions take a callback argument to signal completion
    def first(c, complete):
        def end_first():
            c.s.state = 'second'     
            print('first completed')

            # completes the action after 3 seconds
            complete(None)
        
        start_timer(3, end_first)
        
    @when_all(s.state == 'second')
    def second(c, complete):
        def end_second():
            c.s.state = 'third'
            print('second completed')

            # completes the action after 6 seconds
            # use the first argument to signal an error
            complete(Exception('error detected'))

        start_timer(6, end_second)

        # overrides the 5 second default abandon timeout
        return 10
    
update_state('flow', { 'state': 'first' })

```

</div>
</details>

<details>
<summary>Unhandled Exceptions</summary>
<div markdown="1">

* 액션에서 예외가 처리되지 않은 경우, 예외는 컨텍스트 상태에 저장됩니다. 
* 이를 통해 예외 처리 규칙을 작성할 수 있습니다.

``` python

from durable.lang import *

with ruleset('flow'):
    
    @when_all(m.action == 'start')
    def first(c):
        raise Exception('Unhandled Exception!')

    # when the exception property exists
    @when_all(+s.exception)
    def second(c):
        print(c.s.exception)
        c.s.exception = None
            
post('flow', { 'action': 'start' })

```
</div>
</details>

# Flow Structures

<details>
<summary>Statechart</summary>
<div markdown="1">

* 규칙은 상태도(statecharts)를 사용하여 구성할 수 있습니다. 상태도는 결정적 유한 오토마타(DFA)입니다. 상태 컨텍스트는 가능한 여러 상태 중 하나에 있으며, 이러한 상태 간에 조건부 전환을 가집니다.

* 상태도 규칙:

1. 상태도는 하나 이상의 상태를 가질 수 있습니다.
2. 상태도에는 초기 상태가 필요합니다.
3. 초기 상태는 들어오는 간선이 없는 정점으로 정의됩니다.
4. 상태는 0개 이상의 트리거를 가질 수 있습니다.
5. 상태는 0개 이상의 상태를 가질 수 있습니다 (중첩 상태 참조).
6. 트리거에는 목적지 상태가 있습니다.
7. 트리거는 규칙을 가질 수 있습니다 (부재는 상태 진입을 의미).
8. 트리거는 액션을 가질 수 있습니다.

``` python

from durable.lang import *

with statechart('expense'):
    # initial state 'input' with two triggers
    with state('input'):
        # trigger to move to 'denied' given a condition
        @to('denied')
        @when_all((m.subject == 'approve') & (m.amount > 1000))
        # action executed before state change
        def denied(c):
            print ('denied amount {0}'.format(c.m.amount))
        
        @to('pending')    
        @when_all((m.subject == 'approve') & (m.amount <= 1000))
        def request(c):
            print ('requesting approve amount {0}'.format(c.m.amount))
    
    # intermediate state 'pending' with two triggers
    with state('pending'):
        @to('approved')
        @when_all(m.subject == 'approved')
        def approved(c):
            print ('expense approved')
            
        @to('denied')
        @when_all(m.subject == 'denied')
        def denied(c):
            print ('expense denied')
    
    # 'denied' and 'approved' are final states    
    state('denied')
    state('approved')
        
# events directed to default statechart instance
post('expense', { 'subject': 'approve', 'amount': 100 })
post('expense', { 'subject': 'approved' })

# events directed to statechart instance with id '1'
post('expense', { 'sid': 1, 'subject': 'approve', 'amount': 100 })
post('expense', { 'sid': 1, 'subject': 'denied' })

# events directed to statechart instance with id '2'
post('expense', { 'sid': 2, 'subject': 'approve', 'amount': 10000 })

```

</div>
</details>

<details>
<summary>Nested States</summary>
<div markdown="1">

* 중첩 상태를 사용하면 컴팩트한 상태도를 작성할 수 있습니다. 
* 컨텍스트가 중첩 상태에 있는 경우, 컨텍스트는 묵시적으로 주변 상태에도 있습니다. 
* 상태도는 하위 상태 컨텍스트에서 모든 이벤트를 처리하려고 시도합니다. * 하위 상태가 이벤트를 처리하지 않으면, 이벤트는 자동으로 상위 상태 컨텍스트에서 처리됩니다.

``` python

from durable.lang import *

with statechart('worker'):
    # super-state 'work' has two states and one trigger
    with state('work'):
        # sub-state 'enter' has only one trigger
        with state('enter'):
            @to('process')
            @when_all(m.subject == 'enter')
            def continue_process(c):
                print('start process')
    
        with state('process'):
            @to('process')
            @when_all(m.subject == 'continue')
            def continue_process(c):
                print('continue processing')

        # the super-state trigger will be evaluated for all sub-state triggers
        @to('canceled')
        @when_all(m.subject == 'cancel')
        def cancel(c):
            print('cancel process')

    state('canceled')

# will move the statechart to the 'work.process' sub-state
post('worker', { 'subject': 'enter' })

# will keep the statechart to the 'work.process' sub-state
post('worker', { 'subject': 'continue' })
post('worker', { 'subject': 'continue' })

# will move the statechart out of the work state
post('worker', { 'subject': 'cancel' })

```

</div>
</details>

<details>
<summary>Flowchart</summary>
<div markdown="1">

* 플로우차트는 규칙 세트 흐름을 구성하는 또 다른 방법입니다. 플로우차트에서 각 단계는 실행할 액션을 나타냅니다. 따라서 (상태도 상태와 달리) 컨텍스트 상태에 적용되면 다른 단계로 전환됩니다.

* 플로우차트 규칙:

1. 플로우차트는 하나 이상의 단계를 가질 수 있습니다.
2. 플로우차트에는 초기 단계가 필요합니다.
3. 초기 단계는 들어오는 간선이 없는 정점으로 정의됩니다.
4. 단계는 액션을 가질 수 있습니다.
5. 단계는 0개 이상의 조건을 가질 수 있습니다.
6. 조건에는 규칙과 목적지 단계가 있습니다.

``` python

from durable.lang import *

with flowchart('expense'):
    # initial stage 'input' has two conditions
    with stage('input'): 
        to('request').when_all((m.subject == 'approve') & (m.amount <= 1000))
        to('deny').when_all((m.subject == 'approve') & (m.amount > 1000))
    
    # intermediate stage 'request' has an action and three conditions
    with stage('request'):
        @run
        def request(c):
            print('requesting approve')
            
        to('approve').when_all(m.subject == 'approved')
        to('deny').when_all(m.subject == 'denied')
        # reflexive condition: if met, returns to the same stage
        to('request').when_all(m.subject == 'retry')
    
    with stage('approve'):
        @run 
        def approved(c):
            print('expense approved')

    with stage('deny'):
        @run
        def denied(c):
            print('expense denied')

```

``` python

# events for the default flowchart instance, approved after retry
post('expense', { 'subject': 'approve', 'amount': 100 })
# post('expense', { 'subject': 'retry' })
# post('expense', { 'subject': 'approved' })

# # events for the flowchart instance '1', denied after first try
# post('expense', { 'sid': 1, 'subject': 'approve', 'amount': 100})
# post('expense', { 'sid': 1, 'subject': 'denied'})

# # # event for the flowchart instance '2' immediately denied
# post('expense', { 'sid': 2, 'subject': 'approve', 'amount': 10000})

```

</div>
</details>

<details>
<summary>Timer</summary>
<div markdown="1">

* 이벤트는 타이머를 사용하여 예약할 수 있습니다. 
* 시간 초과 조건은 규칙 전제에 포함될 수 있습니다.
* 기본적으로 타임아웃은 이벤트로 트리거됩니다 (한 번만 관찰됨).
* '수동 리셋' 타이머에 의해 타임아웃은 사실로도 트리거될 수 있으며, 액션 실행 중 타이머를 리셋할 수 있습니다 (마지막 예제 참조).

``` planetext

start_timer: 지정된 이름과 지속 시간으로 타이머를 시작합니다 (manual_reset은 선택 사항입니다).
reset_timer: '수동 리셋' 타이머를 리셋합니다.
cancel_timer: 진행 중인 타이머를 취소합니다.
timeout: 전제 조건으로 사용됩니다.
from durable.lang import *

```

``` python

with ruleset('timer'):
    
    @when_all(m.subject == 'start')
    def start(c):
        c.start_timer('MyTimer', 5)
        
    @when_all(timeout('MyTimer'))
    def timer(c):
        print('timer timeout')

post('timer', { 'subject': 'start' })

```

* 아래 예제에서는 타이머를 사용하여 더 높은 이벤트 비율을 감지합니다.

``` python

from durable.lang import *

with statechart('risk'):
    with state('start'):
        @to('meter')
        def start(c):
            c.start_timer('RiskTimer', 5)

    with state('meter'):
        @to('fraud')
        @when_all(count(3), c.message << m.amount > 100)
        def fraud(c):
            for e in c.m:
                print(e.message) 

        @to('exit')
        @when_all(timeout('RiskTimer'))
        def exit(c):
            print('exit')

    state('fraud')
    state('exit')

```

``` python

# three events in a row will trigger the fraud rule
post('risk', { 'amount': 200 })
post('risk', { 'amount': 300 })
post('risk', { 'amount': 400 })

# two events will exit after 5 seconds
post('risk', { 'sid': 1, 'amount': 500 })
post('risk', { 'sid': 1, 'amount': 600 })

```

* 이 예제에서는 속도를 측정하기 위해 수동 리셋 타이머를 사용합니다.

``` python

from durable.lang import *

with statechart('risk'):
    with state('start'):
        @to('meter')
        def start(c):
            c.start_timer('VelocityTimer', 5, True)

    with state('meter'):
        @to('meter')
        @when_all(cap(5), 
                  m.amount > 100,
                  timeout('VelocityTimer'))
        def some_events(c):
            print('velocity: {0} in 5 seconds'.format(len(c.m)))
            # resets and restarts the manual reset timer
            c.reset_timer('VelocityTimer')
            c.start_timer('VelocityTimer', 5, True)

        @to('meter')
        @when_all(pri(1), timeout('VelocityTimer'))
        def no_events(c):
            print('velocity: no events in 5 seconds')
            c.reset_timer('VelocityTimer')
            c.start_timer('VelocityTimer', 5, True)

```

``` planetext

post('risk', { 'amount': 200 })
post('risk', { 'amount': 300 })
post('risk', { 'amount': 50 })
post('risk', { 'amount': 500 })
post('risk', { 'amount': 600 })

```


</div>
</details>

</div>
</details>

### 참고[¶]()

- 산업인공지능개론 과목, 이건명 교수
- https://github.com/jruizgit/rules

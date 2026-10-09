from pathlib import Path
from unittest.mock import patch
from tests.test_ice_checkpoint_mirror import MODULE
import pytest

GOOD="Disk quotas for usr tlertuss3 (uid 3559929):\n /storage/ice1 267224804 314572800 314572800 - 817591 1000000 1000000 -\n"

def parse(text=GOOD):
    return MODULE.parse_lustre_quota(text,Path('/storage/ice1'),'tlertuss3',3559929)

def test_native_bytes_inode_and_authority():
    proof=parse(); assert proof['available_bytes']==47347996*1024
    assert proof['available_inodes']==182409
    assert proof['quota_bytes']==314572800*1024
    assert proof['usage_authority']=='lfs quota -u'

@pytest.mark.parametrize('text',[GOOD.replace('tlertuss3','other'),GOOD.replace('3559929','1'),GOOD.replace('314572800','0'),GOOD.replace('1000000','0'),GOOD+GOOD.split('\n')[1]+'\n',GOOD.replace('267224804','bad')])
def test_invalid_identity_limits_or_counters_rejected(text):
    with pytest.raises(ValueError): parse(text)

def test_soft_limit_and_overage_conservative():
    proof=parse(GOOD.replace('267224804','400000000*').replace('314572800 314572800','300000000 314572800'))
    assert proof['available_bytes']==0
    assert proof['quota_bytes']==300000000*1024

def test_filesystem_mismatch_rejected_before_query(tmp_path):
    with patch.object(MODULE.os,'stat') as stat, patch.object(MODULE,'run_checked') as command:
        stat.side_effect=[type('S',(),{'st_dev':1})(),type('S',(),{'st_dev':2})()]
        with pytest.raises(ValueError): MODULE.lustre_quota(tmp_path,tmp_path,2)
        command.assert_not_called()

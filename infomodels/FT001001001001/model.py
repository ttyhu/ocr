import re
from full_ocr_local import full_ocr


class FT001001001001:
    """
    营业执照
    """

    def __init__(self):
        """
        初始化文件和正则模式
        """
        self.text = ''
        self.filter_char = ['】', '【', '^', '”', '“', '_']

        self.code_mode = [re.compile('代码([0-9A-Za-z]+)')]  # 统一社会信用代码正则模式
        self.no_mode = [re.compile('(?:册号|津号|注号)(.*)')]  # 注册号正则模式
        self.company_mode = [re.compile('(?:名.{0,1}称)(.*)')]  # 企业名称正则模式
        self.reg_money_mode = [re.compile('(?:册资本|册咨.|注册资.)(.*)')]  # 注册资本正则模式
        self.rea_money_mode = [re.compile('(?:收资本|收咨.)(.*)')]  # 实收资本正则模式
        self.address_mode = [re.compile('(?:住.{0,1}[所己])(.*?[镇省市路街道号县].*)')]  # 住所正则模式
        self.lawyer_mode = [re.compile('(?:姓名)(.*)'), re.compile('(?:代表人)(.*)')]  # 法定代表人正则模式
        self.type_mode = [re.compile('(?:类.{0,1}型|目米刑)(.*)')]  # 公司类型正则模式
        self.scope_mode = [re.compile('(?:[范茫]围)(.*)')]  # 经营范围正则模式
        self.life_mode = [re.compile('(?:业.{0,1}期.{0,1}限)(.*)')]  # 营业期限正则模式
        self.est_date_mode = [re.compile('(?:立.{0,1}日.{0,1}期|成.{0,1}立.{0,1}日.)(.*)')]  # 成立日期正则模式
        self.all_life_mode = [re.compile('伙期限(.*)')]  # 合伙期限正则模式
        self.office_mode = [re.compile('哈哈哈(.*)')]  # 发证机关正则模式
        self.date_mode = [re.compile('(\w{0,4}年\w{0,2}月\w{0,3}日)')]  # 发证日期正则模式

        self.code = ''  # 统一社会信用代码'
        self.no = ''  # 注册号
        self.company = ''  # 企业名称
        self.reg_money = ''  # 注册资本
        self.rea_money = ''  # 实收资本
        self.address = ''  # 住所
        self.lawyer = ''  # 法定代表人
        self.type = ''  # 公司类型
        self.scope = ''  # 经营范围
        self.life = ''  # 营业期限
        self.est_date = ''  # 成立日期
        self.all_life = ''  # 合伙期限
        self.office = ''  # 发证机关
        self.date = ''  # 发证日期

    def extract_info(self, file_path, page, FT):
        """
        提取文档信息
        :param file_path: 文件路径
        :return: dict
        """
        self.text = full_ocr(file_path, page, FT)  # 将docx文件转化为文本
        print(self.text)

        self.code = self.extract_code()  # 统一社会信用代码
        self.no = self.extract_no()  # 注册号
        self.company = self.extract_company()  # 企业名称
        self.reg_money = self.extract_reg_money()  # 注册资本
        self.rea_money = self.extract_rea_money()  # 实收资本
        self.address = self.extract_address()  # 住所
        self.lawyer = self.extract_lawyer()  # 法定代表人
        self.type = self.extract_type()  # 公司类型
        self.scope = self.extract_scope()  # 经营范围
        self.life = self.extract_life()  # 营业期限
        self.est_date = self.extract_est_date()  # 成立日期
        self.all_life = self.extract_all_life()  # 合伙期限
        self.office = self.extract_office()  # 发证机关
        self.date = self.extract_date()  # 发证日期

        if self.company.endswith('公'):
            self.company += '司'

        return {'证书名称': '营业执照', '所属主体名称': self.company.split(' ')[0], '统一社会信用代码': self.code.split(' ')[0], '注册号': self.no.split(' ')[0],
                '企业名称': self.company.split(' ')[0], '注册资本': self.reg_money, '实收资本': self.rea_money,
                '住所': self.address.split(' ')[0], '法定代表人': self.lawyer, '公司类型': self.type.split(' ')[0],
                '经营范围': self.scope.split(' ')[0], '营业期限': self.life, '成立日期': self.est_date, '合伙期限': self.all_life,
                '发证机关': self.office, '发证日期': self.date}

    def extract_company(self):
        """
         提取注册号
        :return: str（注册号）
        """
        value = ''
        for line in self.text.split():
            v = self.company_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_code(self):
        """
         提取类别
        :return: str（类别）
        """
        value = ''
        for line in self.text.split():
            v = self.code_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_no(self):
        """
         提取注册人
        :return: str（注册人）
        """
        value = ''
        for line in self.text.split():
            v = self.no_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_reg_money(self):
        """
         提取注册日期
        :return: str（注册日期）
        """
        value = ''
        for line in self.text.split():
            v = self.reg_money_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_rea_money(self):
        """
         提取有效期
        :return: str（有效期）
        """
        value = ''
        for line in self.text.split():
            v = self.rea_money_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_address(self):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for line in self.text.split():
            v = self.address_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_lawyer(self):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for line in self.text.split():
            for i in self.lawyer_mode:
                v = i.search(''.join([i for i in line if i not in self.filter_char]))
                if v:
                    value = v.groups()[0]
                    break
        return value

    def extract_type(self):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for line in self.text.split():
            v = self.type_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_scope(self):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for line in self.text.split():
            v = self.scope_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_life(self):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for line in self.text.split():
            v = self.life_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_est_date(self):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for line in self.text.split():
            v = self.est_date_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_all_life(self):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for line in self.text.split():
            v = self.all_life_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_office(self):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for line in self.text.split():
            v = self.office_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value

    def extract_date(self):
        """
         提取注册有效期限
        :return: str（注册有效期限）
        """
        value = ''
        for line in self.text.split():
            v = self.date_mode[0].search(''.join([i for i in line if i not in self.filter_char]))
            if v:
                value = v.groups()[0]
                break
        return value